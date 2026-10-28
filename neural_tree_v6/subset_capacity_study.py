"""Resumable, separate CIFAR correction-capacity follow-up. Earlier runs are inputs.

The root is frozen; every learned unit is an actual positive/negative correction
child. Model widths, not new flat classifiers, vary. Test results never schedule
jobs or choose repeat widths.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'neural_tree_v6'))
from common import *
from data import load_bundle, test_features
from run import fit_job, model_folder, labels_for_job, keep_awake
from train import BudgetExpired, evaluate, restore_forest


def subset_bundle(config, folder):
    source = ROOT / 'neural_tree_v6/runs/augmentation-study/features/cifar10'
    bundle = load_bundle(source, config)
    labels = bundle['splits']['train']['y']
    generator = torch.Generator().manual_seed(730410)
    indices = torch.cat([torch.where(labels == c)[0][torch.randperm(int((labels == c).sum()), generator=generator)[:400]]
                         for c in range(10)])
    assert len(indices) == 4000 and len(indices.unique()) == 4000
    old = bundle['splits']['train']
    bundle = {**bundle, 'splits': {**bundle['splits']},
              'signature': digest({'parent': bundle['signature'], 'indices': tensor_hash(indices)}),
              'train_views': bundle['train_views'][:, indices].clone(), 'debug_subset': False}
    bundle['splits']['train'] = {'x': old['x'][indices].clone(), 'y': old['y'][indices].clone(),
                                'ids': [old['ids'][i] for i in indices.tolist()]}
    noisy = labels_for_job(bundle, {'noise': .15, 'dataset': 'cifar10'}, config)
    fixed_json(folder / 'subset-manifest.json', {
        'selection_seed': 730410, 'source_indices': indices.tolist(),
        'ids': bundle['splits']['train']['ids'], 'clean_labels': labels[indices].tolist(),
        'training_labels': noisy.tolist(), 'changed_labels': int((noisy != labels[indices]).sum()),
        'feature_signature': bundle['signature'], 'feature_views_sha256': [
            file_hash(source / f'train-view-{v}.pt') for v in range(3)],
        'scope': '4,000 distinct original images; CNN previously trained on full clean training pool. Not a full-data fit.'})
    return bundle


def main(hours):
    folder = ROOT / 'neural_tree_v6/runs/noisy-subset-capacity'
    output = ROOT / 'neural_tree_v6/results/noisy-subset-capacity'
    folder.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    config = {**read_json(ROOT / 'neural_tree_v6/config.json'),
              'capacity_epochs': 1000, 'capacity_depth': 1, 'checkpoint_every': 20}
    protocol = {
        'dataset': 'cifar10', 'originals': 4000, 'noise': .15, 'epochs_per_child': 1000,
        'root': 'unchanged frozen CNN80 margins; no root MLP is trained',
        'topology': 'depth one: ten OVR roots, up to twenty correction MLPs; inactive positions retained in allocation',
        'primary_width_order': [32, 128, 512, 1024, 2048, 8, 16, 64, 256],
        'arms': ['none', 'augmented'], 'primary_seed': 409, 'repeat_seeds': [410, 411],
        'training_seconds_cap': None if hours is None else hours * 3600,
        'extension': 'After primary widths, add 4096 then 8192 per arm only until three tested widths interpolate, time permitting.',
        'repeat_selection': 'Training-only: largest noninterpolating width, first interpolating width, largest completed width; low/middle/high if none fit. Two additional optimizer seeds.',
        'test_policy': 'All completed, fixed or training-selected jobs evaluated after the queue finishes or hits its time cap; previously used test split is exploratory.',
        'regularization': 'No dropout, weight decay or smoothing. Fixed original plus two crop/flip views; noisy label shared by views.',
        'scope': 'Standalone subset diagnostic, not a replacement for 210 completed full-data configurations.',
    }
    definition = {'protocol': protocol, 'config': config,
               'source_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), *sorted((ROOT/'neural_tree_v6').glob('*.py'))]}}
    definition_path = folder / 'definition.json'
    if hours is None and definition_path.exists():
        # A time-policy amendment must not change any fitting identity or source.
        original = read_json(definition_path)
        runner_key = str(Path(__file__).relative_to(ROOT))
        for value in [original, definition]:
            assert value['config'] == config
        original_study = {k:v for k,v in original['protocol'].items() if k!='training_seconds_cap'}
        resumed_study = {k:v for k,v in protocol.items() if k!='training_seconds_cap'}
        assert original_study == resumed_study, 'Scientific protocol changed.'
        assert {k:v for k,v in original['source_hashes'].items() if k!=runner_key} == {
            k:v for k,v in definition['source_hashes'].items() if k!=runner_key}, 'Training source changed.'
        bounded_source = ROOT/'output/training_tools/history/subset_capacity_study-bounded.py'
        assert file_hash(bounded_source) == original['source_hashes'][runner_key], 'Original launcher backup changed.'
        fixed_json(folder/'unlimited-resume-definition.json', {
            'authorization': 'User requested resume with no upper time limit on 23 October 2026.',
            'original_definition_sha256': file_hash(definition_path), 'runtime_definition': definition,
            'changes': 'Wall-clock cap removed; checkpoint identities, optimizer settings, data and fitting sources unchanged. Evaluation plans retained by phase.'})
    else:
        fixed_json(definition_path, definition)
    state = {'pid': os.getpid(), 'status': 'starting', 'started_unix': time.time(), 'completed': 0,
             'planned_primary': 18, 'time_cap_hours': hours, 'test_evaluated': False}
    started = time.monotonic(); deadline = float('inf') if hours is None else started + hours * 3600
    def progress(**fields):
        state.update(fields); state['updated_unix'] = time.time()
        write_json(folder / 'progress.json', state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_jobs = []
    def job(width, arm, seed=409):
        return {'dataset': 'cifar10', 'augmentation': arm, 'noise': .15, 'width': width, 'depth': 1,
                'epochs': 1000, 'sampler': 'full', 'sample_fraction': 1., 'replicate': 0, 'seed': seed}
    primary = [job(w, arm) for w in protocol['primary_width_order'] for arm in protocol['arms']]
    fixed_json(folder / 'primary-plan.json', primary)
    def run_one(j, stage):
        if j not in all_jobs: all_jobs.append(j)
        write_json(folder / 'scheduled-plan.json', all_jobs)
        progress(status='running', stage=stage, job=j)
        record = fit_job(folder, output, bundle, j, config, device, progress, deadline)
        path = model_folder(folder, j) / 'model.pt'
        with zipfile.ZipFile(path) as archive: assert archive.testzip() is None
        complete = [read_json(model_folder(folder, k)/'metrics.json') for k in all_jobs
                    if (model_folder(folder, k)/'metrics.json').exists()]
        write_json(output/'measurements.json', complete)
        progress(status='running', stage='completed_model', completed=len(complete), last_completed_job=j,
                 elapsed_seconds=time.monotonic()-started)
        if len(complete) == 2:
            seconds = sum(r['seconds_this_invocation'] for r in complete)
            write_json(output/'benchmark.json', {'two_model_seconds': seconds,
                'remaining_primary_estimate_seconds_at_same_speed': seconds/2*16,
                'qualification': 'Widths and interpolation difficulty change runtime; extrapolation is provisional.'})
        return record
    with study_lock(folder/'runner.lock'), keep_awake() as awake:
        progress(temporary_system_awake_request=awake, device=str(device), stage='preparing')
        try:
            bundle = subset_bundle(config, folder)
            for j in primary: run_one(j, 'primary_training')
            for arm in protocol['arms']:
                for w in [4096, 8192]:
                    rows = [read_json(model_folder(folder, j)/'metrics.json') for j in all_jobs
                            if j['augmentation']==arm and j['seed']==409]
                    if sum(r['train_views']['errors']==0 for r in rows)>=3: break
                    decision={'width':w,'arm':arm,'zero_error_widths_before':sum(r['train_views']['errors']==0 for r in rows),
                              'selection_uses':'training-view errors only'}
                    fixed_json(folder/f'extension-{arm}-{w}.json',decision)
                    run_one(job(w,arm),'training_selected_extension')
            repeats=[]
            for arm in protocol['arms']:
                rows=sorted([read_json(model_folder(folder,j)/'metrics.json') for j in all_jobs
                             if j['augmentation']==arm and j['seed']==409],key=lambda r:r['job']['width'])
                first=next((i for i,r in enumerate(rows) if r['train_views']['errors']==0),None)
                selected=([max(0,first-1),first,len(rows)-1] if first is not None
                          else [0,len(rows)//2,len(rows)-1])
                for i in sorted(set(selected)):
                    for seed in [410,411]: repeats.append(job(rows[i]['job']['width'],arm,seed))
            fixed_json(folder/'repeat-plan.json',repeats)
            for j in repeats: run_one(j,'training_selected_repeats')
            progress(training_status='complete')
        except (BudgetExpired,TimeoutError):
            progress(training_status='time_limit',stage='training_stopped_checkpoints_preserved')
            print('Training time cap reached; evaluating completed models only.',flush=True)
        except BaseException as exc:
            progress(status='failed',error=repr(exc));raise
        progress(stage='final_test_evaluation',status='evaluating')
        completed=[j for j in all_jobs if (model_folder(folder,j)/'metrics.json').exists()]
        evaluation_plan={'jobs':completed,'incomplete_excluded':[j for j in all_jobs if j not in completed],
                   'policy':'Frozen before loading test features. No model selection from test.'}
        evaluation_path=folder/'evaluation-plan.json'
        if evaluation_path.exists() and read_json(evaluation_path)!=evaluation_plan:
            evaluation_path=folder/('evaluation-plan-'+digest(evaluation_plan)[:12]+'.json')
        fixed_json(evaluation_path,evaluation_plan)
        if completed:
            heldout=test_features('cifar10',bundle,config['feature_source'])
            for j in completed:
                destination=model_folder(folder,j);record=read_json(destination/'metrics.json')
                if 'test' not in record:
                    forest,_=restore_forest(destination/'model.pt',device)
                    record['test'],p,pred=evaluate(forest,heldout['x'],heldout['y'],config['eval_batch'])
                    record['test_status']='exploratory_reused_test_after_fixed_or_training_selected_queue'
                    saved=load_checkpoint(destination/'predictions.pt');saved.update(test_probabilities=p,test_prediction=pred,test_y=heldout['y'])
                    save_checkpoint(destination/'predictions.pt',saved);write_json(destination/'metrics.json',record)
            write_json(output/'measurements.json',[read_json(model_folder(folder,j)/'metrics.json') for j in completed])
        progress(status='complete',stage='complete',completed=len(completed),test_evaluated=True,
                 elapsed_seconds=time.monotonic()-started)
        print(f'Finished: {len(completed)} complete configurations, all evaluated. Earlier runs unchanged.',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    cap=parser.add_mutually_exclusive_group()
    cap.add_argument('--hours',type=float,default=4.25)
    cap.add_argument('--no-time-limit',action='store_true')
    args=parser.parse_args()
    if not args.no_time_limit and not 0<args.hours<=4.5: parser.error('Use a cap between zero and 4.5 hours, or --no-time-limit.')
    main(None if args.no_time_limit else args.hours)
