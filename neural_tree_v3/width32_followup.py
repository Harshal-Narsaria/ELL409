"""Preserved-CNN width-32 extension, with a separate classification refit arm."""
from pathlib import Path
import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch
import study as s

PROJECT = Path(__file__).resolve().parent


def classification_refit(inputs, labels):
    """Find the largest feasible margin up to one, then economical signed weights.

    The original unit-margin LP can fail even when all labels can be classified
    correctly. This LP leaves the base-score coefficient fixed at one, and keeps
    positive/negative child coefficients nonnegative. It never reads held-out data.
    The returned float32 predictions are verified rather than trusting LP status.
    """
    from scipy.optimize import linprog
    values = inputs.detach().cpu().double().numpy()
    targets = labels.detach().cpu().numpy()
    rows, bounds = [], []
    for label in (0, 1):
        sign = 2 * label - 1
        for a in (0, 1):
            for b in (0, 1):
                mask = (targets == label) & (values[:, 1] == a) & (values[:, 2] == b)
                if mask.any():
                    rows.append([-sign*a, sign*b, -sign, 1.])
                    bounds.append(float(np.min(sign*values[mask, 0])))
    options = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
    maximum = linprog([0., 0., 0., -1.], A_ub=rows, b_ub=bounds,
        bounds=[(0, None), (0, None), (None, None), (0, 1)], method='highs', options=options)
    metadata = {'classification_feasible': False, 'maximum_margin_up_to_one': None}
    if not maximum.success or maximum.x[3] <= 1e-8:
        return None, metadata
    margin = float(maximum.x[3])
    # Half the feasible margin provides room for floating-point rounding.
    economical = linprog([1., 1., 0.], A_ub=np.asarray(rows)[:, :3],
        b_ub=np.asarray(bounds)-.5*margin,
        bounds=[(0, None), (0, None), (None, None)], method='highs', options=options)
    if not economical.success:
        return None, metadata
    weights = torch.tensor(economical.x, dtype=inputs.dtype, device=inputs.device)
    prediction = inputs[:, 0]+weights[0]*inputs[:, 1]-weights[1]*inputs[:, 2]+weights[2]
    metadata.update(maximum_margin_up_to_one=margin, verified_errors=s.score_metrics(prediction, labels)[0])
    if metadata['verified_errors'] != 0:
        return None, metadata
    metadata['classification_feasible'] = True
    return weights, metadata


def refit(inputs, labels, path, epochs, identity, policy):
    if policy == 'original':
        return s.fit_correction(inputs, labels, path, epochs, identity)
    # When exact classification is feasible, avoid unnecessary Adam fitting.
    exact, metadata = classification_refit(inputs, labels)
    if exact is not None:
        return exact, {'classification_lp': metadata, 'method': 'verified-positive-margin-lp'}
    weights, stats = s.fit_correction(inputs, labels, path, epochs, identity)
    return weights, {**stats, 'classification_lp': metadata, 'method': 'original-fallback'}


class FollowupTrainer(s.ProgressiveTrainer):
    def __init__(self, *args, policy, correction_folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy, self.correction_folder = policy, Path(correction_folder)

    def grow(self, indices, y, depth_limit, node_cap, path='root', depth=0):
        self.unvisited_existing.discard(path)
        self.used += 1
        s.seed_everything(self.config['seed']+self.class_index+int(s.digest(path)[:7], 16))
        node = s.FeatureNode(128, self.config['width'], 128+self.root_index if depth == 0 else None).to(self.images.device)
        identity = {'path': path, 'depth': depth, 'kind': 'mlp', 'class': self.class_index, 'pixels': self.fingerprint}
        base_config = {k: self.config[k] for k in ('width', 'seed', 'mlp_epochs', 'mlp_batch', 'cnn_epochs', 'cnn_batch', 'lr', 'eval_batch')}
        if depth:
            stats = s.fit_node(node.base, self.images, indices, y, self.folder/'bases'/(path+'.pt'),
                base_config, identity, lambda **fields:self.progress(class_index=self.class_index, **fields))
        else:
            stats = {'frozen_cnn_root': True, 'history': []}
        base_scores = s.scores_on(node.base, self.images, indices, self.config['eval_batch'])
        before = s.score_metrics(base_scores, y)
        self.tasks[path] = {'base': stats, 'examples': len(y), 'errors_before': before[0]}
        if depth < depth_limit and before[0]:
            for side, (targets, mask) in s.correction_targets(base_scores >= 0, y).items():
                child_path = path+'.'+side
                has_room = child_path in self.unvisited_existing or self.used+len(self.unvisited_existing) < node_cap
                if has_room and targets[mask].sum() > 0:
                    setattr(node, side, self.grow(indices[mask], targets[mask], depth_limit, node_cap, child_path, depth+1))
        if node.A is not None or node.B is not None:
            a = (s.scores_on(node.A, self.images, indices, self.config['eval_batch']) >= 0).float() if node.A is not None else torch.zeros_like(y)
            b = (s.scores_on(node.B, self.images, indices, self.config['eval_batch']) >= 0).float() if node.B is not None else torch.zeros_like(y)
            weights, stats = refit(torch.stack([base_scores, a, b], 1), y,
                self.correction_folder/(path+'.pt'), self.config['correction_epochs'], identity, self.policy)
            with torch.no_grad():
                node.alpha.copy_(weights[0]); node.beta.copy_(weights[1]); node.offset.copy_(weights[2])
            after = s.score_metrics(s.scores_on(node, self.images, indices, self.config['eval_batch']), y)
            if after > before:
                with torch.no_grad(): node.alpha.zero_(); node.beta.zero_(); node.offset.zero_()
            self.tasks[path]['correction'] = stats
        self.tasks[path]['errors_after'] = s.score_metrics(s.scores_on(node, self.images, indices, self.config['eval_batch']), y)[0]
        return node


def saturated(saved, cap):
    """No further base-error descendants can be added, independently of test scores."""
    for tasks in saved['tasks'].values():
        if len(tasks) >= cap:
            return False
        for path, task in tasks.items():
            if not any(p.startswith(path+'.') for p in tasks) and task['errors_before']:
                return False
    return True


def checkpoint_path(run, source, policy, seed, depth):
    if policy == 'original' and seed == 409 and depth <= 10:
        return source/f'seed-{seed}'/'width-32'/f'depth-{depth}'/'forest.pt'
    return run/policy/f'seed-{seed}'/f'depth-{depth}'/'forest.pt'


def fit_stage(run, source, bundle, config, policy, seed, depth, device, progress):
    destination = run/policy/f'seed-{seed}'/f'depth-{depth}'
    if (destination/'metrics.json').exists():
        return s.read_json(destination/'metrics.json')
    if policy == 'original' and seed == 409 and depth <= 10:
        record = s.read_json(source/f'seed-{seed}'/'width-32'/f'depth-{depth}'/'metrics.json')
        record.update(policy=policy, reused_original_stage=True)
        s.write_json(destination/'metrics.json', record)
        return record
    local = {**config, 'seed': seed, 'width': 32, 'cnn_epochs': config['cnn_max_epochs']}
    classes = bundle['classes']; count = 1 if classes == 2 else classes
    cap = min(config['max_nodes_per_tree'], max(1, config['max_correction_parameters']//count//(130*32+4)))
    earlier = sorted([p for p in destination.parent.glob('depth-*/metrics.json') if int(p.parent.name.split('-')[1]) < depth],
        key=lambda p:int(p.parent.name.split('-')[1]))
    previous = s.read_json(earlier[-1]) if earlier else None
    previous_saved = s.load_checkpoint(checkpoint_path(run, source, policy, seed, previous['depth_limit'])) if previous else None
    if previous_saved and saturated(previous_saved, cap):
        record = {**previous, 'depth_limit': depth, 'reused_saturated_model': True,
                  'saturated_from_depth': previous.get('saturated_from_depth', previous['depth_limit'])}
        s.save_checkpoint(destination/'forest.pt', previous_saved)
        s.write_json(destination/'metrics.json', record)
        print(f'{policy} seed {seed} depth {depth}: saturated; no additional capacity', flush=True)
        return record
    x, y = bundle['splits']['train']['x'].to(device), bundle['splits']['train']['y'].to(device)
    roots, tasks = [], {}
    for i, c in enumerate([1] if classes == 2 else range(classes)):
        old_root = s.FeatureNode.from_spec(previous_saved['spec']['roots'][i]) if previous_saved else None
        existing = [path for path, _, _ in old_root.walk()] if old_root else []
        trainer = FollowupTrainer(run/'shared-bases'/f'seed-{seed}'/f'class-{c}', local, x,
            bundle['signature'], c, i, progress, existing, policy=policy,
            correction_folder=destination/f'class-{c}')
        roots.append(trainer.grow(torch.arange(len(y), device=device), (y == c).float(), depth, cap).cpu())
        tasks[str(c)] = trainer.tasks
    forest = s.FeatureForest(classes, roots).to(device)
    record = {'seed': seed, 'width': 32, 'depth_limit': depth, 'policy': policy,
              'node_cap_per_tree': cap, 'size': forest.size(), 'feature_signature': bundle['signature'],
              'selection_uses_test': False, 'test_repeated_exploratory': True}
    for split, data in bundle['splits'].items():
        record[split] = s.evaluate(forest, data['x'].to(device), data['y'].to(device), config['eval_batch'])
    record['convergence_flags'] = [{'class':c, 'node':path, 'last_errors':t['base']['history'][-1]['errors']}
        for c, ts in tasks.items() for path, t in ts.items() if t['base']['history'] and
        t['base']['history'][-1]['errors'] > 0 and len(t['base']['history']) >= 5 and
        t['base']['history'][-1]['loss'] < .99*t['base']['history'][-5]['loss']]
    s.save_checkpoint(destination/'forest.pt', {'spec':forest.spec(), 'model':s.cpu_state(forest), 'tasks':tasks})
    s.write_json(destination/'metrics.json', record)
    print(f"{policy} seed {seed} depth {depth}: train {record['train']['accuracy']:.3%}, validation {record['validation']['accuracy']:.3%}, test {record['test']['accuracy']:.3%}, nodes {record['size']['nodes']}", flush=True)
    return record


def export(run, output, config, cnn_parameters):
    records = []
    for path in run.glob('*/seed-*/depth-*/metrics.json'):
        row = s.read_json(path)
        row['total_parameters'] = cnn_parameters+row['size']['parameters']
        records.append(row)
    records.sort(key=lambda r:(r['policy'], r['seed'], r['depth_limit']))
    s.write_json(output/'measurements.json', records)
    fields = ['policy', 'seed', 'depth_limit', 'total_parameters', 'nodes', 'train_accuracy', 'validation_accuracy', 'test_accuracy', 'saturated']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'measurements.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({**{k:r[k] for k in fields[:4]}, 'nodes':r['size']['nodes'],
                **{split+'_accuracy':r[split]['accuracy'] for split in ('train','validation','test')},
                'saturated':r.get('reused_saturated_model',False)})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for policy in config['policies']:
        rows = [r for r in records if r['policy']==policy and r['seed']==config['seed']]
        # One point per actual primary-seed checkpoint, not per saturated depth limit.
        rows = [r for r in rows if not r.get('reused_saturated_model')]
        for axis, split in zip(axes, ('train','validation','test')):
            axis.plot([r['total_parameters'] for r in rows], [100*(1-r[split]['accuracy']) for r in rows],
                'o-', label=policy.replace('_',' '))
            zero = next((r for r in rows if r['train']['errors']==0), None)
            if zero:
                axis.scatter([zero['total_parameters']], [100*(1-zero[split]['accuracy'])], marker='*', s=170, color='black', zorder=5)
            axis.set(xscale='log', xlabel='Actual total parameters (CNN counted once)', ylabel=f'{split.capitalize()} error (%)')
            axis.grid(alpha=.2); axis.legend()
    fig.suptitle('CIFAR-10 width 32: original depth extension vs separate parent refit; stars = first interpolation')
    fig.tight_layout(); fig.savefig(output/'parameter-error.png', dpi=170); plt.close(fig)
    observations = {}
    for policy in config['policies']:
        primary = [r for r in records if r['policy']==policy and r['seed']==config['seed']]
        observations[policy] = {'first_zero_training_error_depth': next((r['depth_limit'] for r in primary if r['train']['errors']==0), None),
            'actual_capacity_plateau': any(r.get('reused_saturated_model',False) for r in primary)}
    s.write_json(output/'capacity-observations.json', {'exploratory_test_informed_followup':True,
        'saturation_is_not_additional_capacity':True, 'policies':observations})
    summary=[]
    for policy in config['policies']:
        for depth in config['repeat_depths']:
            rows=[r for r in records if r['policy']==policy and r['depth_limit']==depth]
            if len(rows)<2: continue
            row={'policy':policy,'depth_limit':depth,'seeds':[r['seed'] for r in rows],
                 'parameters_min':min(r['total_parameters'] for r in rows),
                 'parameters_max':max(r['total_parameters'] for r in rows)}
            for split in ('train','validation','test'):
                errors=np.asarray([1-r[split]['accuracy'] for r in rows])
                row[split]={'mean_error':float(errors.mean()),'standard_error':float(errors.std(ddof=1)/np.sqrt(len(errors)))}
            summary.append(row)
    s.write_json(output/'seed-summary.json', {'conditional_on_one_frozen_cnn':True,
        'depth_limits_can_have_same_actual_capacity':True,'records':summary})
    if summary:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5))
        for policy in config['policies']:
            rows=[r for r in summary if r['policy']==policy]
            for axis,split in zip(axes,('train','validation','test')):
                axis.errorbar([r['depth_limit'] for r in rows], [100*r[split]['mean_error'] for r in rows],
                    yerr=[100*r[split]['standard_error'] for r in rows],fmt='o-',capsize=4,label=policy.replace('_',' '))
                axis.set(xlabel='Maximum depth (may have identical capacity)',ylabel=f'{split.capitalize()} error (%)')
                axis.grid(alpha=.2); axis.legend()
        fig.suptitle('Width 32 repeats: mean ± standard error over available correction seeds; one frozen CNN')
        fig.tight_layout(); fig.savefig(output/'repeat-error.png',dpi=170); plt.close(fig)
    return records


def run(config_path, run_dir, output, device):
    followup = s.read_json(config_path)
    config = s.read_json(PROJECT/followup['source_config'])
    source = (PROJECT/followup['source_run']).resolve()
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if run_dir == source or source in run_dir.parents:
        raise ValueError('Follow-up must use a separate run directory')
    for protected in (s.WORKSPACE/'neural_tree', PROJECT/'runs/cnn80-depth', PROJECT/'results/cnn80-depth'):
        if any(p == protected or protected in p.parents for p in (run_dir, output)):
            raise ValueError('Existing project/results are protected')
    status = {'status':'preparing', 'pid':os.getpid(), 'updated_unix':time.time()}
    last_write = [0.]
    def progress(**fields):
        status.update(fields); status['updated_unix']=time.time()
        if 'epoch' in fields and time.time()-last_write[0]<5: return
        try: s.write_json(run_dir/'progress.json',status); last_write[0]=time.time()
        except PermissionError: pass
    with s.study_lock(run_dir/'runner.lock'):
        code = [Path(__file__), PROJECT/'study.py', s.WORKSPACE/'neural_tree/ell409_tree/models.py']+list((s.WORKSPACE/'neural_tree_v2/v2_tree').glob('*.py'))
        protected_files = [source/'features.pt', source/'cnn-frozen.pt', source/'pixels.pt']
        protected_files += list((source/f"seed-{followup['seed']}"/'width-32').rglob('*.pt'))
        protected_files += list((source/f"seed-{followup['seed']}"/'width-32').rglob('*.json'))
        protected_files += list((PROJECT/'results/cnn80-depth/cifar10').glob('*.*'))
        s.fixed_definition(run_dir/'definition.json', {'followup':followup, 'source_config':config,
            'code':{str(p):s.file_hash(p) for p in code}, 'source_hashes':{str(p):s.file_hash(p) for p in protected_files}})
        definition = s.read_json(run_dir/'definition.json')
        try:
            s.seed_everything(followup['seed'])
            bundle = s.load_checkpoint(source/'features.pt')
            if bundle['classes'] != 10 or followup['width'] != 32:
                raise ValueError('This follow-up is specifically CIFAR-10 width 32')
            cnn_parameters = s.read_json(PROJECT/'results/cnn80-depth/cifar10/cnn-baseline.json')['parameters']
            # Copy trusted source base checkpoints, including optimizer/RNG state, once.
            for path in (source/f"seed-{followup['seed']}"/'width-32').glob('class-*/bases/*.pt'):
                target = run_dir/'shared-bases'/f"seed-{followup['seed']}"/path.relative_to(source/f"seed-{followup['seed']}"/'width-32')
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
            jobs = [(followup['seed'], followup['depths'])]+[(seed,followup['repeat_depths']) for seed in followup['repeat_seeds']]
            for seed, depths in jobs:
                for policy in followup['policies']:
                    for depth in depths:
                        progress(status='running',seed=seed,policy=policy,width=32,depth_limit=depth)
                        fit_stage(run_dir,source,bundle,config,policy,seed,depth,device,progress)
                        records = export(run_dir,output,followup,cnn_parameters)
                        progress(completed_records=len(records))
            # Select by full-training fit and size only, independently for each arm.
            for policy in followup['policies']:
                primary = [r for r in records if r['policy']==policy and r['seed']==followup['seed']]
                fitting = [r for r in primary if r['train']['accuracy']>=config['near_perfect']]
                selected = min(fitting,key=lambda r:(r['total_parameters'],r['depth_limit'])) if fitting else max(primary,key=lambda r:(r['train']['accuracy'],-r['total_parameters']))
                forest_saved = s.load_checkpoint(checkpoint_path(run_dir,source,policy,selected['seed'],selected['depth_limit']))
                frozen, pixels = s.load_checkpoint(source/'cnn-frozen.pt'), s.load_checkpoint(source/'pixels.pt')
                s.save_checkpoint(output/policy/'deployment.pt', {'cnn_model':frozen['model'], 'forest_spec':forest_saved['spec'],
                    'forest_model':forest_saved['model'], 'image_mean':pixels['mean'], 'image_std':pixels['std'],
                    'feature_mean':bundle['feature_mean'], 'feature_std':bundle['feature_std'], 'classes':pixels['classes'],
                    'resolution':config['resolution'], 'selection':selected, 'cnn_checkpoint_sha256':s.file_hash(source/'cnn-frozen.pt')})
                s.write_json(output/policy/'selection.json', {'selection':selected,'uses_test':False,'training_target_met':bool(fitting)})
            s.verify_originals(definition['source_hashes'])
            progress(status='complete',source_results_verified_unchanged=True)
        except Exception as exc:
            progress(status='failed',error=repr(exc)); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'width32-followup.json'))
    parser.add_argument('--run-dir',default=str(PROJECT/'runs/width32-followup'))
    parser.add_argument('--output',default=str(PROJECT/'results/width32-followup/cifar10'))
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args()
    run(args.config,args.run_dir,args.output,torch.device(args.device))


if __name__=='__main__': main()
