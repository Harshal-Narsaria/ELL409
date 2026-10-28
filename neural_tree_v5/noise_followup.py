"""Continue exact matching node tasks, recomputing children after root fitting."""
from pathlib import Path
import argparse
from contextlib import contextmanager
import gc
import os
import sys
import time
import noise_study as s
import torch

PROJECT, WORKSPACE = s.PROJECT, s.WORKSPACE


def fork_checkpoint(source, destination, indices, labels, local, identity, old_budget):
    """Rewrite only a validated epoch-budget signature into a separate checkout.

    A changed correction target is a different optimization problem. Its old
    optimizer/history/best-key must never be silently reused.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists(): return {'mode': 'resume-followup'}
    if not source.exists(): return {'mode': 'fresh', 'reason': 'no-original-checkpoint'}
    old_local = {**local, 'root_epochs': old_budget['root_epochs'], 'mlp_epochs': old_budget['mlp_epochs']}
    task = s.tensor_hash(indices, labels)
    expected = s.digest({'identity': identity, 'config': old_local, 'task': task})
    saved = s.load_checkpoint(source)
    if saved['signature'] != expected:
        return {'mode': 'fresh', 'reason': 'training-task-or-configuration-changed'}
    if local['root_epochs'] < old_local['root_epochs'] or local['mlp_epochs'] < old_local['mlp_epochs']:
        raise ValueError('Continuation budget cannot decrease.')
    metadata = {'mode': 'continued-exact-task', 'source_checkpoint': str(source),
        'source_sha256': s.file_hash(source), 'starting_epoch': saved['epoch'],
        'source_signature': saved['signature'], 'task_hash': task}
    saved['signature'] = s.digest({'identity': identity, 'config': local, 'task': task})
    saved['continuation_source'] = metadata
    s.save_checkpoint(destination, saved)
    return metadata


@contextmanager
def continue_matching_nodes(old_folder, new_folder, old_config):
    original = s.fit_node
    old_folder, new_folder = Path(old_folder).resolve(), Path(new_folder).resolve()
    def fit(model, images, indices, labels, path, local, identity, progress=None):
        relative = Path(path).resolve().relative_to(new_folder)
        metadata = fork_checkpoint(old_folder/relative, path, indices, labels, local, identity, old_config)
        audit = Path(path).with_suffix('.continuation.json')
        if not audit.exists(): s.write_json(audit, metadata)
        return original(model, images, indices, labels, path, local, identity, progress)
    s.fit_node = fit
    try: yield
    finally: s.fit_node = original


def collect_summary(run_dir, output, job, base_config):
    rows = []
    for dataset in job['datasets']:
        for rate in job['noise_rates']:
            for seed in job['seeds']:
                for width in job['widths_by_dataset'][dataset]:
                    path = run_dir/dataset/s.noise_key(rate)/f'seed-{seed}'/f'width-{width}'/'metrics.json'
                    if path.exists(): rows.append({'dataset': dataset, **s.read_json(path)})
    s.write_json(output/'summary.json', {'planned_models': sum(len(job['widths_by_dataset'][d])*len(job['noise_rates'])*len(job['seeds']) for d in job['datasets']),
        'completed_models': len(rows), 'records': rows,
        'interpretation': 'Compare common-budget widths and optimizer seeds; no automatic double-descent conclusion. Test partitions were reused. Seeds are conditional on one noise map and one CNN per dataset.'})


def run(config_path, run_dir, output, device):
    job, run_dir, output = s.read_json(config_path), Path(run_dir), Path(output)
    base = s.read_json(WORKSPACE/job['source_config'])
    old_run = WORKSPACE/job['source_run']
    if s.read_json(old_run/'progress.json')['status'] != 'complete':
        raise RuntimeError('Original noise study must be complete.')
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
    state = {'pid': os.getpid(), 'status': 'starting', 'planned_models': sum(len(job['widths_by_dataset'][d])*len(job['noise_rates'])*len(job['seeds']) for d in job['datasets'])}
    def progress(**fields):
        state.update(fields); state['updated_unix'] = time.time()
        s.write_json(run_dir/'progress.json', state)
    with s.study_lock(run_dir/'runner.lock'):
        with s.study_lock(old_run/'runner.lock'): pass
        s.fixed_definition(run_dir/'definition.json', {'job': job, 'base_config': base,
            'implementation': {name: s.file_hash(PROJECT/name) for name in ('noise_study.py', 'noise_followup.py')}})
        if not (run_dir/'preserved-originals.json').exists():
            snapshot = s.preservation_snapshot(base)
            snapshot.update({str(p): s.file_hash(p) for p in old_run.rglob('*') if p.is_file() and p.suffix in ('.pt', '.json')})
            s.write_json(run_dir/'preserved-originals.json', snapshot)
        snapshot = s.read_json(run_dir/'preserved-originals.json'); s.verify_originals(snapshot)
        s.write_json(run_dir/'environment.json', {'python': sys.version, 'torch': torch.__version__, 'device': str(device),
            'GPU': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
        try:
            completed = len(list(run_dir.glob('*/noise-*/seed-*/width-*/metrics.json')))
            for dataset in job['datasets']:
                progress(status='loading_frozen_features', dataset=dataset, completed_models=completed)
                source, metadata = s.source_bundle(dataset, base)
                s.write_json(output/dataset/'feature-extractor.json', metadata)
                cfg = {**base, 'datasets': [dataset], 'widths': job['widths_by_dataset'][dataset],
                    'root_epochs': job['root_epochs'], 'mlp_epochs': job['mlp_epochs'],
                    'seed': job['seeds'][0], 'repeat_seeds': job['seeds'][1:],
                    'protocol': job['policy']}
                for rate in job['noise_rates']:
                    folder = run_dir/dataset/s.noise_key(rate)
                    bundle = s.noisy_bundle(source, rate, base['noise_seed'], folder)
                    original_manifest = s.read_json(old_run/dataset/s.noise_key(rate)/'noise-manifest.json')
                    assert original_manifest == s.read_json(folder/'noise-manifest.json'), 'Noise map changed.'
                    s.fixed_definition(folder/'repeat-plan.json', {'widths': cfg['widths'], 'seeds': cfg['repeat_seeds'],
                        'uses_training_only': True, 'predeclared_followup': True})
                    with continue_matching_nodes(old_run/dataset/s.noise_key(rate), folder, base):
                        for seed in job['seeds']:
                            for width in cfg['widths']:
                                progress(status='longer_training', noise_fraction=rate, seed=seed, width=width)
                                s.fit_stage(folder, bundle, cfg, seed, width, progress, device, metadata['cnn_parameters'])
                                completed = len(list(run_dir.glob('*/noise-*/seed-*/width-*/metrics.json')))
                                progress(completed_models=completed)
                                s.export_results(run_dir/dataset, output/dataset, cfg)
                                collect_summary(run_dir, output, job, cfg)
                    del bundle; gc.collect()
                progress(status='final_test_evaluation')
                for rate in job['noise_rates']:
                    folder = run_dir/dataset/s.noise_key(rate)
                    bundle = s.noisy_bundle(source, rate, base['noise_seed'], folder)
                    s.finalize_test(folder, bundle, cfg, device)
                s.export_results(run_dir/dataset, output/dataset, cfg)
                collect_summary(run_dir, output, job, cfg)
                s.verify_originals(snapshot)
                del source, bundle; gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
            progress(status='complete', previous_results_verified_unchanged=True, completed_models=completed)
        except BaseException as exc:
            progress(status='failed', error=repr(exc)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT/'followup-config.json'))
    parser.add_argument('--run-dir', default=str(PROJECT/'runs/noise-followup'))
    parser.add_argument('--output', default=str(PROJECT/'results/noise-followup'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    run(args.config, args.run_dir, args.output, torch.device(args.device))


if __name__ == '__main__': main()
