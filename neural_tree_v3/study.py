"""Frozen 80%-accuracy CNN roots with progressively deeper correction trees."""
from pathlib import Path
import argparse
import csv
import gc
import os
import sys
import time

PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'runs/matplotlib-cache'))
sys.path.insert(0, str(WORKSPACE / 'neural_tree_v2'))
sys.path.insert(0, str(WORKSPACE / 'neural_tree'))
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from ell409_tree.models import SharedCNN
from v2_tree.common import (cpu_state, digest, file_hash, load_checkpoint, read_json,
    restore_rng, rng_state, save_checkpoint, seed_everything, study_lock, tensor_hash, write_json)
from v2_tree.data import make_cache, normalized
from v2_tree.models import ImageForest, correction_targets
from v2_tree.study import fixed_definition, protected_originals, verify_originals
from v2_tree.train import evaluate, fit_correction, fit_node, score_metrics, scores_on


def root_scores(logits):
    """Winner-vs-strongest-alternative margins preserve CNN argmax decisions."""
    if logits.shape[1] == 2:
        difference = logits[:, 1] - logits[:, 0]
        # CNN argmax chooses class 0 on ties, whereas the tree's >=0 rule chooses 1.
        return torch.where(difference == 0, torch.full_like(difference, -torch.finfo(difference.dtype).tiny), difference)[:, None]
    alternatives = logits[:, None, :].expand(-1, logits.shape[1], -1).clone()
    alternatives[:, torch.arange(logits.shape[1]), torch.arange(logits.shape[1])] = -torch.inf
    return logits - alternatives.amax(2)


class FeatureScore(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.features = features
        self.layers = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.layers(x[:, :self.features]).squeeze(1)


class FrozenRootScore(nn.Module):
    def __init__(self, column):
        super().__init__()
        self.column = column

    def forward(self, x):
        return x[:, self.column]


class FeatureNode(nn.Module):
    def __init__(self, features, width, root_column=None):
        super().__init__()
        self.features, self.width, self.root_column = features, width, root_column
        self.base = FrozenRootScore(root_column) if root_column is not None else FeatureScore(features, width)
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.beta = nn.Parameter(torch.tensor(0.))
        self.offset = nn.Parameter(torch.tensor(0.))
        self.A = self.B = None

    def forward(self, x):
        score = self.base(x) + self.offset
        if self.A is not None:
            score = score + self.alpha.clamp_min(0) * (self.A(x) >= 0).to(x.dtype)
        if self.B is not None:
            score = score - self.beta.clamp_min(0) * (self.B(x) >= 0).to(x.dtype)
        return score

    def walk(self, path='root', depth=0):
        yield path, depth, self
        for side in ('A', 'B'):
            if getattr(self, side) is not None:
                yield from getattr(self, side).walk(path + '.' + side, depth + 1)

    def spec(self):
        return {'features': self.features, 'width': self.width, 'root_column': self.root_column,
                'A': self.A.spec() if self.A is not None else None,
                'B': self.B.spec() if self.B is not None else None}

    @classmethod
    def from_spec(cls, spec):
        node = cls(spec['features'], spec['width'], spec['root_column'])
        for side in ('A', 'B'):
            if spec[side] is not None: setattr(node, side, cls.from_spec(spec[side]))
        return node


class FeatureForest(ImageForest):
    @classmethod
    def from_spec(cls, spec):
        return cls(spec['classes'], [FeatureNode.from_spec(s) for s in spec['roots']])

    def size(self):
        walked = [row for root in self.roots for row in root.walk()]
        return {'parameters': sum(p.numel() for p in self.parameters()), 'nodes': len(walked),
                'max_depth': max(d for _, d, _ in walked),
                'correction_nodes': sum(n.root_column is None for _, _, n in walked)}


class ImageClassifier(nn.Module):
    """Deploy one frozen CNN, then its feature-based correction forest."""
    def __init__(self, cnn, forest, image_mean, image_std, feature_mean, feature_std):
        super().__init__()
        self.cnn, self.forest = cnn, forest
        for p in cnn.parameters(): p.requires_grad_(False)
        for name, value in [('image_mean', image_mean), ('image_std', image_std),
                            ('feature_mean', feature_mean), ('feature_std', feature_std)]:
            self.register_buffer(name, value.clone())

    def forward(self, image):
        image = (image - self.image_mean[None, :, None, None]) / self.image_std[None, :, None, None]
        z = self.cnn.features(image)
        x = torch.cat([(z - self.feature_mean) / self.feature_std, root_scores(self.cnn.head(z))], 1)
        return self.forest(x)


def load_deployment(path, device='cpu'):
    saved = load_checkpoint(path)
    cnn = SharedCNN(saved['forest_spec']['classes'])
    cnn.load_state_dict(saved['cnn_model'])
    forest = FeatureForest.from_spec(saved['forest_spec'])
    forest.load_state_dict(saved['forest_model'])
    model = ImageClassifier(cnn, forest, saved['image_mean'], saved['image_std'],
                            saved['feature_mean'], saved['feature_std'])
    return model.to(device).eval(), saved


def cnn_metrics(model, images, labels, batch):
    errors, loss = 0, 0.
    model.eval()
    with torch.no_grad():
        for x, y in zip(images.split(batch), labels.split(batch)):
            logits = model(x)
            errors += int((logits.argmax(1) != y).sum())
            loss += float(F.cross_entropy(logits, y, reduction='sum'))
    return {'accuracy': 1-errors/len(labels), 'errors': errors, 'examples': len(labels), 'loss': loss/len(labels)}


def train_to_target(model, images, labels, folder, config, fingerprint, progress):
    signature = digest({'data': fingerprint, 'seed': config['seed'], 'target': config['cnn_target_accuracy'],
        'epochs': config['cnn_max_epochs'], 'lr': config['cnn_lr'], 'batch': config['cnn_batch']})
    latest, frozen = folder/'cnn-latest.pt', folder/'cnn-frozen.pt'
    if frozen.exists():
        saved = load_checkpoint(frozen)
        if saved['signature'] != signature: raise ValueError('Frozen CNN identity changed')
        model.load_state_dict(saved['model'])
        for p in model.parameters(): p.requires_grad_(False)
        model.eval()
        return saved
    optimizer = torch.optim.Adam(model.parameters(), lr=config['cnn_lr'], weight_decay=0)
    history, start = [], 0
    if latest.exists():
        saved = load_checkpoint(latest)
        if saved['signature'] != signature: raise ValueError('CNN resume identity changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']); history, start = saved['history'], saved['epoch']
    # If interruption occurred between the latest commit and the frozen commit, freeze without another update.
    reached = bool(history and history[-1]['train']['accuracy'] >= config['cnn_target_accuracy'])
    for epoch in range(start, config['cnn_max_epochs']):
        if reached: break
        before = time.perf_counter()
        model.train()
        order = torch.randperm(len(labels), device=images.device)
        for positions in order.split(config['cnn_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images[positions]), labels[positions], label_smoothing=0)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite CNN loss')
            loss.backward(); optimizer.step()
        tr = cnn_metrics(model, images, labels, config['cnn_batch'])
        history.append({'epoch': epoch+1, 'train': tr, 'seconds': time.perf_counter()-before})
        saved = {'signature': signature, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                 'rng': rng_state(), 'history': history, 'epoch': epoch+1}
        save_checkpoint(latest, saved)
        progress(status='training_cnn', epoch=epoch+1, training_accuracy=tr['accuracy'])
        print(f"CNN epoch {epoch+1}: training {tr['accuracy']:.3%}", flush=True)
        reached = tr['accuracy'] >= config['cnn_target_accuracy']
    if not reached:
        raise RuntimeError('CNN did not reach the declared training target; no frozen-80% result claimed.')
    save_checkpoint(frozen, saved)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return saved


def test_pixels(manifest_path, resolution, limit=None):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    records = manifest['splits']['test']
    if limit is not None: records = records[:limit]
    arrays = np.load(manifest_path.parent/'images.npy', mmap_mode='r') if (manifest_path.parent/'images.npy').exists() else None
    pixels = torch.empty(len(records), 3, resolution, resolution, dtype=torch.uint8)
    for i, record in enumerate(records):
        if 'index' in record:
            image = Image.fromarray(arrays[record['index']])
        else:
            with Image.open(manifest_path.parent/record['path']) as source: image = source.convert('RGB')
        image = image.convert('RGB').resize((resolution, resolution), Image.Resampling.BILINEAR)
        pixels[i] = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    return {'pixels': pixels, 'y': torch.tensor([r['label'] for r in records]), 'ids': [r['id'] for r in records]}


def extract_features(cnn, cache, folder, checkpoint_hash, device):
    path = folder/'features.pt'
    signature = digest({'pixels': cache['fingerprint'], 'cnn': checkpoint_hash,
                        'test': tensor_hash(cache['splits']['test']['pixels'], cache['splits']['test']['y'])})
    if path.exists():
        bundle = load_checkpoint(path)
        if bundle['signature'] != signature: raise ValueError('Feature cache identity changed')
        return bundle
    bundle = {'signature': signature, 'classes': len(cache['classes']), 'splits': {}}
    cnn.eval()
    with torch.no_grad():
        for split, data in cache['splits'].items():
            features, margins = [], []
            for pixels in data['pixels'].split(128):
                z = cnn.features(normalized(pixels, cache, device))
                features.append(z.cpu()); margins.append(root_scores(cnn.head(z)).cpu())
            bundle['splits'][split] = {'z': torch.cat(features), 'margins': torch.cat(margins), 'y': data['y'], 'ids': data['ids']}
    z = bundle['splits']['train']['z']
    bundle['feature_mean'], bundle['feature_std'] = z.mean(0), z.std(0, unbiased=False).clamp_min(1e-6)
    for data in bundle['splits'].values():
        data['x'] = torch.cat([(data.pop('z')-bundle['feature_mean'])/bundle['feature_std'], data.pop('margins')], 1)
    save_checkpoint(path, bundle)
    return bundle


class ProgressiveTrainer:
    def __init__(self, folder, config, images, fingerprint, class_index, root_index, progress, existing_paths=()):
        self.folder, self.config, self.images = Path(folder), config, images
        self.fingerprint, self.class_index, self.root_index = fingerprint, class_index, root_index
        self.progress, self.used, self.tasks = progress, 0, {}
        self.unvisited_existing = set(existing_paths)

    def grow(self, indices, y, depth_limit, node_cap, path='root', depth=0):
        self.unvisited_existing.discard(path)
        self.used += 1
        seed_everything(self.config['seed'] + self.class_index + int(digest(path)[:7], 16))
        node = FeatureNode(128, self.config['width'], 128+self.root_index if depth == 0 else None).to(self.images.device)
        # Base identity deliberately excludes stage depth/cap: its inputs, targets, width and schedule do not change.
        identity = {'path': path, 'depth': depth, 'kind': 'mlp', 'class': self.class_index, 'pixels': self.fingerprint}
        base_config = {k: self.config[k] for k in ('width', 'seed', 'mlp_epochs', 'mlp_batch', 'cnn_epochs', 'cnn_batch', 'lr', 'eval_batch')}
        def progress(**fields): self.progress(class_index=self.class_index, **fields)
        if depth:
            stats = fit_node(node.base, self.images, indices, y, self.folder/'bases'/(path+'.pt'), base_config, identity, progress)
        else:
            stats = {'frozen_cnn_root': True, 'history': []}
        base_scores = scores_on(node.base, self.images, indices, self.config['eval_batch'])
        before = score_metrics(base_scores, y)
        self.tasks[path] = {'base': stats, 'examples': len(y), 'errors_before': before[0]}
        if depth < depth_limit and before[0]:
            for side, (targets, mask) in correction_targets(base_scores >= 0, y).items():
                child_path = path+'.'+side
                has_room = child_path in self.unvisited_existing or self.used+len(self.unvisited_existing) < node_cap
                if has_room and targets[mask].sum() > 0:
                    setattr(node, side, self.grow(indices[mask], targets[mask], depth_limit, node_cap, path+'.'+side, depth+1))
        if node.A is not None or node.B is not None:
            a = (scores_on(node.A, self.images, indices, self.config['eval_batch']) >= 0).float() if node.A is not None else torch.zeros_like(y)
            b = (scores_on(node.B, self.images, indices, self.config['eval_batch']) >= 0).float() if node.B is not None else torch.zeros_like(y)
            inputs = torch.stack([base_scores, a, b], 1)
            weights, stats = fit_correction(inputs, y, self.folder/f'depth-{depth_limit}'/(path+'.pt'), self.config['correction_epochs'], identity)
            with torch.no_grad(): node.alpha.copy_(weights[0]); node.beta.copy_(weights[1]); node.offset.copy_(weights[2])
            if score_metrics(scores_on(node, self.images, indices, self.config['eval_batch']), y) > before:
                with torch.no_grad(): node.alpha.zero_(); node.beta.zero_(); node.offset.zero_()
            self.tasks[path]['correction'] = stats
        self.tasks[path]['errors_after'] = score_metrics(scores_on(node, self.images, indices, self.config['eval_batch']), y)[0]
        return node


def fit_stage(folder, cache, config, seed, width, depth, device, progress):
    destination = folder/f'seed-{seed}'/f'width-{width}'/f'depth-{depth}'
    if (destination/'metrics.json').exists(): return read_json(destination/'metrics.json')
    local = {**config, 'seed': seed, 'width': width, 'cnn_epochs': config['cnn_max_epochs']}
    x, y = cache['splits']['train']['x'].to(device), cache['splits']['train']['y'].to(device)
    roots, tasks = [], {}
    classes = cache['classes']; count = 1 if classes == 2 else classes
    per_node = (128+2)*width+4
    cap = min(config['max_nodes_per_tree'], max(1, config['max_correction_parameters']//count//per_node))
    earlier = sorted([p for p in destination.parent.glob('depth-*/forest.pt') if int(p.parent.name.split('-')[1]) < depth],
                     key=lambda p:int(p.parent.name.split('-')[1]))
    previous_specs = load_checkpoint(earlier[-1])['spec']['roots'] if earlier else None
    def spec_paths(spec, path='root'):
        yield path
        for side in ('A','B'):
            if spec[side] is not None: yield from spec_paths(spec[side],path+'.'+side)
    for i, c in enumerate([1] if classes == 2 else range(classes)):
        existing = list(spec_paths(previous_specs[i])) if previous_specs else []
        trainer = ProgressiveTrainer(folder/f'seed-{seed}'/f'width-{width}'/f'class-{c}', local, x, cache['signature'], c, i, progress, existing)
        roots.append(trainer.grow(torch.arange(len(y), device=device), (y == c).float(), depth, cap).cpu())
        tasks[str(c)] = trainer.tasks
    forest = FeatureForest(classes, roots).to(device)
    record = {'seed': seed, 'width': width, 'depth_limit': depth, 'node_cap_per_tree': cap, 'size': forest.size(),
              'feature_signature': cache['signature'], 'selection_uses_test': False, 'test_repeated_exploratory': True}
    for split, data in cache['splits'].items():
        record[split] = evaluate(forest, data['x'].to(device), data['y'].to(device), config['eval_batch'])
    record['convergence_flags'] = [{'class': c, 'node': path, 'last_errors': t['base']['history'][-1]['errors']}
        for c, local_tasks in tasks.items() for path, t in local_tasks.items() if t['base']['history'] and
        t['base']['history'][-1]['errors'] > 0 and len(t['base']['history']) >= 5 and
        t['base']['history'][-1]['loss'] < .99*t['base']['history'][-5]['loss']]
    save_checkpoint(destination/'forest.pt', {'spec': forest.spec(), 'model': cpu_state(forest), 'tasks': tasks})
    write_json(destination/'metrics.json', record)
    print(f"width {width}, depth {depth}: train {record['train']['accuracy']:.3%}, test {record['test']['accuracy']:.3%}, nodes {record['size']['nodes']}", flush=True)
    return record


def export_results(folder, output, cnn_parameters, config):
    records = []
    for path in sorted(folder.glob('seed-*/width-*/depth-*/metrics.json')):
        record = read_json(path)
        record['cnn_parameters'] = cnn_parameters
        record['total_parameters'] = cnn_parameters+record['size']['parameters']
        record['correction_parameters'] = record['size']['parameters']
        records.append(record)
    write_json(output/'measurements.json', records)
    fields = ['seed', 'width', 'depth_limit', 'total_parameters', 'correction_parameters', 'nodes',
              'train_accuracy', 'validation_accuracy', 'test_accuracy', 'train_error', 'test_error']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'measurements.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({**{key:r[key] for key in ('seed','width','depth_limit','total_parameters','correction_parameters')},
                'nodes':r['size']['nodes'], **{s+'_accuracy':r[s]['accuracy'] for s in ('train','validation','test')},
                'train_error':1-r['train']['accuracy'], 'test_error':1-r['test']['accuracy']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if records:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for width in config['widths']:
            rows = sorted([r for r in records if r['seed'] == config['seed'] and r['width'] == width], key=lambda r:r['depth_limit'])
            if not rows: continue
            for axis, split in zip(axes, ('train','test')):
                line, = axis.plot([r['total_parameters'] for r in rows], [100*(1-r[split]['accuracy']) for r in rows], 'o-', label=f'width {width}')
                first_zero = next((r for r in rows if r['train']['errors']==0),None)
                if first_zero:
                    axis.scatter([first_zero['total_parameters']],[100*(1-first_zero[split]['accuracy'])],
                                 marker='*',s=110,color=line.get_color(),zorder=4)
                axis.set_xscale('log'); axis.set_xlabel('Actual total parameters; frozen CNN counted once')
                axis.set_ylabel(f'{split.capitalize()} error (%)'); axis.grid(alpha=.2); axis.legend()
        fig.suptitle(f"CNN stopped at training target {100*config['cnn_target_accuracy']:.0f}%; deeper correction trees; stars mark first interpolation")
        fig.tight_layout(); fig.savefig(output/'parameter-error.png', dpi=170); plt.close(fig)
    observations = {'interpretation': 'No automatic double-descent claim. Depth-stage paths can plateau in active capacity and must not be treated as increasing-size evidence.',
        'widths': {str(w): {'first_zero_error_depth': next((r['depth_limit'] for r in sorted(records,key=lambda r:r['depth_limit'])
            if r['seed']==config['seed'] and r['width']==w and r['train']['errors']==0), None)} for w in config['widths']}}
    write_json(output/'capacity-observations.json', observations)
    return records


def run(config_path, run_dir, output, device, limit=None):
    config = read_json(config_path); run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    original = WORKSPACE/'neural_tree'
    if original == run_dir or original in run_dir.parents or original == output or original in output.parents:
        raise ValueError('New outputs must remain outside the original project')
    status = {'status':'preparing', 'pid':os.getpid(), 'updated_unix':time.time(), 'uses_test_data':True, 'test_curves_exploratory':True}
    last_write = [0.]
    def progress(**fields):
        status.update(fields); status['updated_unix'] = time.time()
        if 'epoch' in fields and time.time()-last_write[0] < 5: return
        try: write_json(run_dir/'progress.json', status); last_write[0] = time.time()
        except PermissionError: pass
    with study_lock(run_dir/'runner.lock'):
        code_paths = [Path(__file__), WORKSPACE/'neural_tree/ell409_tree/models.py'] + list((WORKSPACE/'neural_tree_v2/v2_tree').glob('*.py'))
        fixed_definition(run_dir/'definition.json', {'config':config,'limit':limit,'code':{str(p):file_hash(p) for p in code_paths},
            'manifests':{ds:file_hash(original/'data'/ds/'manifest.json') for ds in config['datasets']}})
        if not (run_dir/'preserved-originals.json').exists(): write_json(run_dir/'preserved-originals.json', protected_originals(original))
        snapshot = read_json(run_dir/'preserved-originals.json'); verify_originals(snapshot)
        try:
            for dataset in config['datasets']:
                progress(status='preparing_pixels',dataset=dataset)
                folder, destination = run_dir/dataset, output/dataset
                manifest = original/'data'/dataset/'manifest.json'
                cache = make_cache(manifest, folder/'pixels.pt', config['resolution'], limit)
                seed_everything(config['seed'])
                cnn = SharedCNN(len(cache['classes'])).to(device)
                images = normalized(cache['splits']['train']['pixels'], cache, device)
                labels = cache['splits']['train']['y'].to(device)
                frozen = train_to_target(cnn, images, labels, folder, config, cache['fingerprint'], progress)
                del images, labels
                for p in cnn.parameters(): p.requires_grad_(False)
                cnn.eval()
                cnn_hash = file_hash(folder/'cnn-frozen.pt')
                progress(status='extracting_features',cnn_training_accuracy=frozen['history'][-1]['train']['accuracy'])
                # Repeated test access is authorized for this distinct exploratory study, after CNN selection.
                cache['splits']['test'] = test_pixels(manifest, config['resolution'], limit)
                bundle = extract_features(cnn, cache, folder, cnn_hash, device)
                parameters = sum(p.numel() for p in cnn.parameters())
                write_json(destination/'cnn-baseline.json', {'selected_epoch':frozen['epoch'], 'train':frozen['history'][-1]['train'],
                    'parameters':parameters,'cnn_checkpoint_sha256':cnn_hash,'selection_uses_train_only':True,'requested_target':config['cnn_target_accuracy']})
                for width in config['widths']:
                    for depth in config['depths']:
                        progress(status='growing_tree',width=width,depth_limit=depth,seed=config['seed'])
                        fit_stage(folder,bundle,config,config['seed'],width,depth,device,progress)
                        records = export_results(folder,destination,parameters,config)
                        progress(completed_records=len(records))
                # Repeat depths picked only by training error, never by attractive test curves.
                reference_width = config['widths'][len(config['widths'])//2]
                reference = sorted([r for r in records if r['seed']==config['seed'] and r['width']==reference_width], key=lambda r:r['depth_limit'])
                crossing = next((r['depth_limit'] for r in reference if r['train']['errors']==0), None)
                chosen = sorted(set([max(0,crossing-1),crossing,min(max(config['depths']),crossing+2)])) if crossing is not None else [config['depths'][0],config['depths'][len(config['depths'])//2],config['depths'][-1]]
                write_json(folder/'repeat-plan.json',{'width':reference_width,'depths':chosen,'selection_uses_training_only':True,'first_zero_error_depth':crossing})
                for repeat_seed in config['repeat_seeds']:
                    for depth in chosen:
                        progress(status='repeat_training',width=reference_width,depth_limit=depth,seed=repeat_seed)
                        fit_stage(folder,bundle,config,repeat_seed,reference_width,depth,device,progress)
                        records = export_results(folder,destination,parameters,config)
                primary = [r for r in records if r['seed']==config['seed']]
                fitting = [r for r in primary if r['train']['accuracy'] >= config['near_perfect']]
                selected = min(fitting,key=lambda r:r['total_parameters']) if fitting else max(primary,key=lambda r:r['train']['accuracy'])
                checkpoint = folder/f"seed-{selected['seed']}"/f"width-{selected['width']}"/f"depth-{selected['depth_limit']}"/'forest.pt'
                forest_saved = load_checkpoint(checkpoint)
                save_checkpoint(destination/'deployment.pt', {'cnn_model':cpu_state(cnn), 'forest_spec':forest_saved['spec'],
                    'forest_model':forest_saved['model'],'image_mean':cache['mean'],'image_std':cache['std'],
                    'feature_mean':bundle['feature_mean'],'feature_std':bundle['feature_std'],'classes':cache['classes'],
                    'resolution':config['resolution'],'selection':selected,'cnn_checkpoint_sha256':cnn_hash})
                write_json(destination/'selection.json',{'width':selected['width'],'depth':selected['depth_limit'],
                    'selection':'Smallest primary-seed model meeting training target; otherwise best training fit. No test selection.',
                    'training_target_met':bool(fitting)})
                if file_hash(folder/'cnn-frozen.pt') != cnn_hash: raise ValueError('Frozen CNN checkpoint changed')
                verify_originals(snapshot)
                del cache,bundle,cnn; gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
            progress(status='complete',original_results_verified_unchanged=True)
        except Exception as exc:
            progress(status='failed',error=repr(exc)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'config.json'))
    parser.add_argument('--run-dir',default=str(PROJECT/'runs/cnn80-depth'))
    parser.add_argument('--output',default=str(PROJECT/'results/cnn80-depth'))
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    run(args.config,args.run_dir,args.output,torch.device(args.device),args.limit)


if __name__ == '__main__': main()
