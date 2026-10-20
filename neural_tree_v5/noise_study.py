"""A separate controlled training-label-noise study using frozen CNN80 features.

Reuses audited upward correction targets, resumable node optimizers, and signed
parent fitting. Original experiments are read-only inputs, not resume targets.
"""
from pathlib import Path
import argparse
import copy
import csv
import gc
import os
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / 'runs/matplotlib-cache'))
sys.path.insert(0, str(WORKSPACE / 'neural_tree_v3'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import study as shared
from width32_followup import refit
from v2_tree.common import (cpu_state, digest, file_hash, load_checkpoint,
    read_json, save_checkpoint, seed_everything, study_lock, tensor_hash, write_json)
from v2_tree.models import correction_targets
from v2_tree.study import fixed_definition, verify_originals
from v2_tree.train import evaluate, fit_node, score_metrics, scores_on


def noise_key(rate):
    return f'noise-{round(rate * 10000):04d}'


def corrupt_labels(labels, features, classes, rate, seed):
    """Wrong-class replacement once per identical-feature group, shared by runs.

    A local RNG avoids changing optimizer RNG state. Group selection is nested
    across rates; original labels are never modified in place. Every selected
    group gets a uniformly chosen *incorrect* class. Image fractions may differ
    slightly from the requested group fraction when duplicates exist.
    """
    if not 0 <= rate < 1 or classes < 2:
        raise ValueError('Need at least two classes and noise rate in [0, 1).')
    clean = labels.detach().cpu().long()
    if len(clean) == 0 or len(features) != len(clean):
        raise ValueError('Features and nonempty labels must match.')
    if int(clean.min()) < 0 or int(clean.max()) >= classes:
        raise ValueError('Label outside declared classes.')
    _, inverse = np.unique(features.detach().cpu().numpy(), axis=0, return_inverse=True)
    groups = int(inverse.max()) + 1
    minima = np.full(groups, classes, dtype=np.int64)
    maxima = np.full(groups, -1, dtype=np.int64)
    np.minimum.at(minima, inverse, clean.numpy())
    np.maximum.at(maxima, inverse, clean.numpy())
    if not np.array_equal(minima, maxima):
        raise ValueError('Identical frozen features already have conflicting labels.')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(groups, generator=generator)
    shifts = torch.randint(1, classes, (groups,), generator=generator)
    selected = torch.zeros(groups, dtype=torch.bool)
    selected[order[:round(groups * rate)]] = True
    group_labels = torch.from_numpy(minima)
    altered = torch.where(selected, (group_labels + shifts) % classes, group_labels)
    noisy = altered[torch.from_numpy(inverse)]
    changed = noisy != clean
    return noisy, {'requested_fraction': rate, 'actual_fraction': float(changed.float().mean()),
        'changed_examples': int(changed.sum()), 'examples': len(clean),
        'unique_feature_groups': groups, 'changed_groups': int(selected.sum()),
        'seed': seed, 'label_hash': tensor_hash(noisy), 'duplicate_groups_consistent': True,
        'original_class_counts': torch.bincount(clean, minlength=classes).tolist(),
        'noisy_class_counts': torch.bincount(noisy, minlength=classes).tolist()}


def noisy_bundle(source, rate, seed, destination):
    """Only the training labels are replaced; keep an explicit original copy."""
    noisy, metadata = corrupt_labels(source['splits']['train']['y'],
        source['splits']['train']['x'], source['classes'], rate, seed)
    identity = {'source_signature': source['signature'], 'rate': rate,
        'noise_seed': seed, 'labels': metadata['label_hash'],
        'ids': digest(source['splits']['train']['ids'])}
    path = Path(destination) / 'noise-manifest.json'
    changes = [{'id': id_, 'original_label': int(c), 'training_label': int(n)}
        for id_, c, n in zip(source['splits']['train']['ids'], source['splits']['train']['y'], noisy)
        if int(c) != int(n)]
    fixed_definition(path, {'identity': identity, 'metadata': metadata, 'changed_samples': changes})
    bundle = {**source, 'signature': digest(identity), 'noise': metadata,
        'clean_train_labels': source['splits']['train']['y'].clone(),
        'splits': {key: {**data} for key, data in source['splits'].items()}}
    bundle['splits']['train']['y'] = noisy
    return bundle


def source_bundle(dataset, config, limit=None):
    folder = WORKSPACE / config['feature_source'] / dataset
    saved = load_checkpoint(folder / 'features.pt')
    manifest = read_json(WORKSPACE / 'neural_tree/data' / dataset / 'manifest.json')
    for key, data in saved['splits'].items():
        expected = manifest['splits'][key]
        if data['ids'] != [r['id'] for r in expected] or not torch.equal(data['y'], torch.tensor([r['label'] for r in expected])):
            raise ValueError('Frozen features do not match the original split manifest.')
    # Deliberately discard CNN head margins: no class scores reach new MLP nodes.
    splits = {key: {'x': data['x'][:, :config['feature_dimension']].contiguous(),
                    'y': data['y'].clone(), 'ids': list(data['ids'])}
              for key, data in saved['splits'].items()}
    if limit is not None:
        splits = {key: {'x': data['x'][:limit], 'y': data['y'][:limit], 'ids': data['ids'][:limit]}
                  for key, data in splits.items()}
    checkpoint = load_checkpoint(folder / 'cnn-frozen.pt')
    cnn = shared.SharedCNN(saved['classes'])
    cnn.load_state_dict(checkpoint['model'])
    # The CNN head is unused and excluded from the deployed parameter count.
    cnn_parameters = sum(p.numel() for p in cnn.blocks.parameters())
    bundle = {'classes': saved['classes'], 'dimension': config['feature_dimension'],
        'signature': digest({'source': saved['signature'], 'dimensions': config['feature_dimension'], 'limit': limit}),
        'splits': splits, 'feature_mean': saved['feature_mean'], 'feature_std': saved['feature_std']}
    metadata = {'feature_source': config['feature_source'], 'cnn_parameters': cnn_parameters,
        'cnn_checkpoint_sha256': file_hash(folder / 'cnn-frozen.pt'),
        'feature_file_sha256': file_hash(folder / 'features.pt'),
        'cnn_original_label_training_accuracy': checkpoint['history'][-1]['train']['accuracy'],
        'cnn_epochs': checkpoint['epoch'], 'feature_dimension': config['feature_dimension'],
        'CNN_head_used': False, 'CNN_trained_before_noise_on_original_training_labels': True,
        'split_counts': {k: len(v['y']) for k, v in splits.items()}, 'debug_subset': limit is not None,
        'test_split_reused_exploratory': True}
    return bundle, metadata


class FixedTrainer:
    def __init__(self, folder, config, x, signature, class_index, seed, width, progress):
        self.folder, self.config, self.x = Path(folder), config, x
        self.signature, self.class_index, self.seed, self.width = signature, class_index, seed, width
        self.progress, self.tasks = progress, {}

    def grow(self, indices, y, depth=0, path='root'):
        seed_everything(self.seed + self.class_index + int(digest(path)[:7], 16))
        node = shared.FeatureNode(self.x.shape[1], self.width).to(self.x.device)
        identity = {'path': path, 'depth': depth, 'kind': 'mlp', 'class': self.class_index,
                    'features_and_labels': self.signature}
        local = {key: self.config[key] for key in ('root_epochs', 'mlp_epochs', 'mlp_batch', 'lr', 'eval_batch')}
        local.update(width=self.width, seed=self.seed)
        if len(y) == 0 or (depth > 0 and int(y.sum()) == 0):
            with torch.no_grad():
                for parameter in node.base.parameters(): parameter.zero_()
                node.base.layers[-1].bias.fill_(-1.)
            stats = {'constant_zero_task': True, 'history': [], 'best_errors': 0, 'elapsed': 0.}
        else:
            stats = fit_node(node.base, self.x, indices, y, self.folder/'bases'/f'class-{self.class_index}'/(path+'.pt'),
                local, identity, lambda **fields: self.progress(class_index=self.class_index, **fields))
        self.tasks[path] = {'base': stats, 'examples': len(y), 'positives': int(y.sum())}
        base_scores = scores_on(node.base, self.x, indices, self.config['eval_batch']) if len(y) else y
        before = score_metrics(base_scores, y) if len(y) else (0, 0.)
        self.tasks[path]['errors_before'] = before[0]
        if depth < self.config['depth']:
            # All positions exist, even when they carry a constant zero task.
            for side, (targets, mask) in correction_targets(base_scores >= 0, y).items():
                setattr(node, side, self.grow(indices[mask], targets[mask], depth+1, path+'.'+side))
            a = (scores_on(node.A, self.x, indices, self.config['eval_batch']) >= 0).float() if len(y) else y
            b = (scores_on(node.B, self.x, indices, self.config['eval_batch']) >= 0).float() if len(y) else y
            if len(y):
                weights, correction = refit(torch.stack([base_scores, a, b], 1), y,
                    self.folder/'corrections'/f'class-{self.class_index}'/(path+'.pt'),
                    self.config['correction_epochs'], identity, 'classification_refit')
                with torch.no_grad():
                    node.alpha.copy_(weights[0]); node.beta.copy_(weights[1]); node.offset.copy_(weights[2])
                if score_metrics(scores_on(node, self.x, indices, self.config['eval_batch']), y) > before:
                    with torch.no_grad(): node.alpha.zero_(); node.beta.zero_(); node.offset.zero_()
                self.tasks[path]['correction'] = correction
        self.tasks[path]['errors_after'] = score_metrics(scores_on(node, self.x, indices, self.config['eval_batch']), y)[0] if len(y) else 0
        return node


def capacity(forest, tasks):
    size = forest.size()
    active = []
    def visit(node):
        active.append(node)
        if node.A is not None and float(node.alpha.detach()) > 0: visit(node.A)
        if node.B is not None and float(node.beta.detach()) > 0: visit(node.B)
    for root in forest.roots: visit(root)
    size['allocated_parameters'] = size['parameters']
    size['active_nodes'] = len(active)
    size['active_parameters'] = sum(sum(p.numel() for p in n.base.parameters()) + 3 for n in active)
    trained = [n for i, root in enumerate(forest.roots) for path, _, n in root.walk()
               if not tasks[str(i)][path]['base'].get('constant_zero_task', False)]
    size['trained_node_parameters'] = sum(sum(p.numel() for p in n.base.parameters()) for n in trained)
    size['trained_nodes'] = len(trained)
    size['parent_fit_parameters'] = 3 * sum(n.A is not None for root in forest.roots for _, _, n in root.walk())
    return size


def roots_only(forest):
    roots = []
    for root in forest.roots:
        node = shared.FeatureNode(root.features, root.width)
        node.base.load_state_dict(cpu_state(root.base))
        roots.append(node)
    return shared.FeatureForest(forest.classes, roots)


def fit_stage(folder, bundle, config, seed, width, progress, device, feature_parameters):
    folder = Path(folder)/f'seed-{seed}'/f'width-{width}'
    identity = digest({'features_and_noise': bundle['signature'], 'config': config, 'seed': seed, 'width': width})
    if (folder/'metrics.json').exists():
        record = read_json(folder/'metrics.json')
        if record['identity'] != identity: raise ValueError('Completed run configuration changed.')
        return record
    start = time.perf_counter()
    x, y = bundle['splits']['train']['x'].to(device), bundle['splits']['train']['y'].to(device)
    roots, tasks = [], {}
    targets = [1] if bundle['classes'] == 2 else list(range(bundle['classes']))
    for i, c in enumerate(targets):
        trainer = FixedTrainer(folder, config, x, bundle['signature'], c, seed, width, progress)
        roots.append(trainer.grow(torch.arange(len(y), device=device), (y == c).float()).cpu())
        tasks[str(i)] = trainer.tasks
    forest = shared.FeatureForest(bundle['classes'], roots).to(device)
    baseline = roots_only(forest).to(device)
    record = {'identity': identity, 'seed': seed, 'width': width, 'depth': config['depth'],
        'noise': bundle['noise'], 'size': capacity(forest, tasks), 'root_size': baseline.size(),
        'feature_parameters': feature_parameters, 'selected_on_training_only': True,
        'test_status': 'withheld_until_predeclared_grid_and_repeats_finish', 'root_only': {}}
    for name in ('train', 'validation'):
        data = bundle['splits'][name]
        record[name] = evaluate(forest, data['x'].to(device), data['y'].to(device), config['eval_batch'])
        record['root_only'][name] = evaluate(baseline, data['x'].to(device), data['y'].to(device), config['eval_batch'])
    record['train_original_labels'] = evaluate(forest, x, bundle['clean_train_labels'].to(device), config['eval_batch'])
    record['root_only']['train_original_labels'] = evaluate(baseline, x, bundle['clean_train_labels'].to(device), config['eval_batch'])
    record['convergence_flags'] = []
    for c, t in tasks.items():
        for path, task in t.items():
            history = task['base']['history']
            if len(history) >= 20 and history[-1]['errors'] > 0 and history[-1]['loss'] < .99*history[-20]['loss']:
                record['convergence_flags'].append({'class': c, 'node': path, 'last_errors': history[-1]['errors'],
                    'last_loss': history[-1]['loss'], 'loss_20_epochs_ago': history[-20]['loss']})
    record['training_seconds'] = sum(t['base'].get('elapsed', 0) + t.get('correction', {}).get('elapsed', 0) for ts in tasks.values() for t in ts.values())
    record['this_invocation_seconds'] = time.perf_counter()-start
    save_checkpoint(folder/'forest.pt', {'identity': identity, 'spec': forest.spec(), 'model': cpu_state(forest), 'tasks': tasks})
    write_json(folder/'metrics.json', record)
    print(f"width {width}, noise {bundle['noise']['actual_fraction']:.2%}: noisy-label train {record['train']['accuracy']:.3%}, original-label train {record['train_original_labels']['accuracy']:.3%}, validation {record['validation']['accuracy']:.3%}", flush=True)
    return record


def representative_widths(records, widths):
    """Below/at/above first zero training error; no held-out score is read."""
    ordered = sorted(records, key=lambda r: r['width'])
    position = next((i for i, r in enumerate(ordered) if r['train']['errors'] == 0), None)
    if position is None:
        return sorted(set([widths[len(widths)//3], widths[2*len(widths)//3], widths[-1]]))
    return sorted(set(ordered[i]['width'] for i in (max(0, position-1), position, min(len(ordered)-1, position+1))))


def finalize_test(folder, bundle, config, device):
    """Unlock test only after the saved primary and repeat protocol is complete."""
    plan = read_json(Path(folder)/'repeat-plan.json')
    expected = [(config['seed'], w) for w in config['widths']] + [(s, w) for s in config['repeat_seeds'] for w in plan['widths']]
    for seed, width in expected:
        if not (Path(folder)/f'seed-{seed}'/f'width-{width}'/'metrics.json').exists():
            raise RuntimeError('Cannot unlock test before all predeclared jobs finish.')
    data = bundle['splits']['test']
    for seed, width in expected:
        stage = Path(folder)/f'seed-{seed}'/f'width-{width}'
        record = read_json(stage/'metrics.json')
        if 'test' in record: continue
        saved = load_checkpoint(stage/'forest.pt')
        forest = shared.FeatureForest.from_spec(saved['spec']).to(device)
        forest.load_state_dict(saved['model'])
        record['test'] = evaluate(forest, data['x'].to(device), data['y'].to(device), config['eval_batch'])
        record['root_only']['test'] = evaluate(roots_only(forest).to(device), data['x'].to(device), data['y'].to(device), config['eval_batch'])
        record['test_status'] = 'evaluated_after_predeclared_jobs; reused_test_split_exploratory'
        write_json(stage/'metrics.json', record)


def export_results(folder, output, config):
    output = Path(output)
    records = [read_json(p) for p in Path(folder).glob('noise-*/seed-*/width-*/metrics.json')]
    records.sort(key=lambda r: (r['noise']['requested_fraction'], r['seed'], r['width']))
    write_json(output/'measurements.json', records)
    output.mkdir(parents=True, exist_ok=True)
    fields = ['noise_fraction', 'seed', 'width', 'depth', 'model', 'allocated_tree_parameters',
        'active_tree_parameters', 'trained_node_parameters', 'total_deployed_parameters',
        'noisy_train_error', 'original_train_error', 'validation_error', 'test_error', 'convergence_flags']
    with (output/'measurements.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in records:
            for model, data, size in [('tree', r, r['size']), ('root-only', r['root_only'], r['root_size'])]:
                writer.writerow({'noise_fraction': r['noise']['actual_fraction'], 'seed': r['seed'], 'width': r['width'],
                    'depth': r['depth'] if model == 'tree' else 0, 'model': model,
                    'allocated_tree_parameters': size['parameters'],
                    'active_tree_parameters': size.get('active_parameters', size['parameters']),
                    'trained_node_parameters': size.get('trained_node_parameters', size['parameters']-3*size['nodes']),
                    'total_deployed_parameters': size['parameters']+r['feature_parameters'],
                    **{key: 1-data[split]['accuracy'] if split in data else '' for key, split in
                       [('noisy_train_error', 'train'), ('original_train_error', 'train_original_labels'), ('validation_error', 'validation'), ('test_error', 'test')]},
                    'convergence_flags': len(r['convergence_flags'])})
    if not records: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for rate in config['noise_rates']:
        rows = [r for r in records if r['noise']['requested_fraction'] == rate and r['seed'] == config['seed']]
        if not rows: continue
        rows.sort(key=lambda r: r['width'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, split in zip(axes.flat, ('train', 'train_original_labels', 'validation', 'test')):
            for model, color in [('root-only', 'tab:blue'), ('tree', 'tab:red')]:
                usable = [r for r in rows if split in (r['root_only'] if model == 'root-only' else r)]
                if not usable: continue
                size_key = 'root_size' if model == 'root-only' else 'size'
                axis.plot([r[size_key]['parameters']+r['feature_parameters'] for r in usable],
                    [100*(1-(r['root_only'] if model == 'root-only' else r)[split]['accuracy']) for r in usable], 'o-', color=color, label=model)
                repeated = [r for r in records if r['noise']['requested_fraction'] == rate and r['seed'] in config['repeat_seeds']+[config['seed']]]
                for width in sorted(set(r['width'] for r in repeated)):
                    group = [r for r in repeated if r['width'] == width and split in (r['root_only'] if model == 'root-only' else r)]
                    if len(group) < 3: continue
                    errors = [100*(1-(r['root_only'] if model == 'root-only' else r)[split]['accuracy']) for r in group]
                    axis.errorbar(group[0][size_key]['parameters']+group[0]['feature_parameters'], np.mean(errors),
                        yerr=np.std(errors, ddof=1)/np.sqrt(len(errors)), fmt='s', color=color, capsize=4)
                first = next((r for r in rows if (r['root_only'] if model == 'root-only' else r)['train']['errors'] == 0), None)
                if first:
                    axis.axvline(first[size_key]['parameters']+first['feature_parameters'], color=color, linestyle='--', alpha=.4)
            axis.set_xscale('log'); axis.set_xlabel('Total deployed parameters (shared frozen CNN included)')
            axis.set_ylabel('Classification error (%)'); axis.grid(alpha=.2)
            axis.set_title({'train': 'Training against study labels', 'train_original_labels': 'Training against original labels',
                'validation': 'Validation: original labels', 'test': 'Test: original labels (exploratory)'}[split])
            if axis.has_data(): axis.legend()
            if split == 'test' and not any('test' in r for r in rows):
                axis.text(.5, .5, 'Withheld until grid and repeats finish', transform=axis.transAxes, ha='center')
        fig.suptitle(f"Training-label noise {rate:.0%}: frozen CNN80 features, fixed tree depth {config['depth']}\nPoints are measured capacities; squares are 3-seed means +/- standard error")
        fig.tight_layout(); fig.savefig(output/(noise_key(rate)+'-capacity.png'), dpi=160); plt.close(fig)
    observations = {}
    for rate in config['noise_rates']:
        primary = sorted([r for r in records if r['seed'] == config['seed'] and r['noise']['requested_fraction'] == rate], key=lambda r: r['width'])
        observations[noise_key(rate)] = {'first_zero_training_error_width': next((r['width'] for r in primary if r['train']['errors'] == 0), None),
            'completed_primary_widths': [r['width'] for r in primary],
            'convergence_flag_count': sum(len(r['convergence_flags']) for r in primary),
            'interpretation': 'Do not infer double descent from a single fluctuation or from allocated inactive positions. Check both descents, interpolation location, actual active capacity and repeats; convergence flags require optimization follow-up.'}
    write_json(output/'capacity-observations.json', observations)


def preservation_snapshot(config):
    paths = []
    for name in ('neural_tree', 'neural_tree_v2', 'neural_tree_v3', 'neural_tree_v4'):
        project = WORKSPACE/name
        paths += [p for p in project.rglob('*.py') if not any(part in ('__pycache__', 'data') for part in p.parts)]
        paths += [p for p in (project/'results').rglob('*') if p.is_file()]
        paths += [p for p in (project/'runs').rglob('metrics.json')]
    paths += [p for p in (WORKSPACE/'output/github-upload/ELL409-final').rglob('*') if p.is_file()]
    for dataset in config['datasets']:
        paths += [WORKSPACE/config['feature_source']/dataset/name for name in ('features.pt', 'cnn-frozen.pt')]
        paths += [WORKSPACE/'neural_tree/data'/dataset/'manifest.json']
    return {str(p): file_hash(p) for p in sorted(set(paths))}


def run(config_path, run_dir, output, device, limit=None):
    config, run_dir, output = read_json(config_path), Path(run_dir), Path(output)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable.')
    if config['depth'] != 1: raise ValueError('This primary study predeclares a fixed depth-one topology.')
    status = {'pid': os.getpid(), 'status': 'starting'}
    def progress(**fields):
        status.update(fields); status['updated_unix'] = time.time()
        write_json(run_dir/'progress.json', status)
    with study_lock(run_dir/'runner.lock'):
        fixed_definition(run_dir/'definition.json', {'config': config, 'limit': limit,
            'source_files': {str(WORKSPACE/config['feature_source']/d/'features.pt'): file_hash(WORKSPACE/config['feature_source']/d/'features.pt') for d in config['datasets']}})
        if not (run_dir/'preserved-originals.json').exists():
            write_json(run_dir/'preserved-originals.json', preservation_snapshot(config))
        preserved = read_json(run_dir/'preserved-originals.json'); verify_originals(preserved)
        write_json(run_dir/'environment.json', {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(),
            'device': str(device), 'GPU': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
        try:
            for dataset in config['datasets']:
                progress(status='loading_fixed_features', dataset=dataset)
                source, metadata = source_bundle(dataset, config, limit)
                write_json(output/dataset/'feature-extractor.json', metadata)
                for rate in config['noise_rates']:
                    folder = run_dir/dataset/noise_key(rate)
                    bundle = noisy_bundle(source, rate, config['noise_seed'], folder)
                    for width in config['widths']:
                        progress(status='primary_sweep', noise_fraction=rate, width=width, seed=config['seed'])
                        fit_stage(folder, bundle, config, config['seed'], width, progress, device, metadata['cnn_parameters'])
                        export_results(run_dir/dataset, output/dataset, config)
                    primary = [read_json(folder/f'seed-{config["seed"]}'/f'width-{w}'/'metrics.json') for w in config['widths']]
                    repeat = representative_widths(primary, config['widths'])
                    fixed_definition(folder/'repeat-plan.json', {'widths': repeat, 'seeds': config['repeat_seeds'],
                        'uses_training_only': True, 'noise_map_shared_across_optimizer_seeds': True})
                    for seed in config['repeat_seeds']:
                        for width in repeat:
                            progress(status='repeat_sweep', noise_fraction=rate, width=width, seed=seed)
                            fit_stage(folder, bundle, config, seed, width, progress, device, metadata['cnn_parameters'])
                            export_results(run_dir/dataset, output/dataset, config)
                    del bundle; gc.collect()
                # No test read until both the original-label control and noise grid finish.
                progress(status='final_test_evaluation')
                for rate in config['noise_rates']:
                    folder = run_dir/dataset/noise_key(rate)
                    bundle = noisy_bundle(source, rate, config['noise_seed'], folder)
                    finalize_test(folder, bundle, config, device)
                export_results(run_dir/dataset, output/dataset, config)
                verify_originals(preserved)
                del source, bundle; gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
            progress(status='complete', previous_results_verified_unchanged=True)
        except BaseException as exc:
            progress(status='failed', error=repr(exc)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT/'config.json'))
    parser.add_argument('--run-dir', default=str(PROJECT/'runs/label-noise'))
    parser.add_argument('--output', default=str(PROJECT/'results/label-noise'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.config, args.run_dir, args.output, torch.device(args.device), args.limit)


if __name__ == '__main__': main()
