"""Complete missing root-only widths within the submission time budget."""
import json
import argparse
import os
from pathlib import Path
import shutil
import time
import torch
import noise_followup as followup

s=followup.s
PROJECT=s.PROJECT
RUN=PROJECT/'runs/root-gap'
OUTPUT=PROJECT/'results/root-gap'


class TimeBudgetReached(Exception):pass


def periodic_checkpoint_writer(original,epochs,interval):
    def write(path,saved):
        epoch=saved.get('epoch')
        if epoch is None or epoch%interval==0 or epoch==epochs:
            return original(path,saved)
    return write


def main(config_path=PROJECT/'root-gap-config.json',run_dir=RUN,output=OUTPUT,
         checkpoint_interval=1,absolute_deadline=None):
    global RUN,OUTPUT
    RUN,OUTPUT=Path(run_dir),Path(output)
    job=s.read_json(config_path)
    base=s.read_json(PROJECT/'config.json')
    start=time.monotonic();deadline=start+job['wall_clock_budget_seconds']
    if job.get('absolute_deadline_unix'):
        deadline=min(deadline,start+max(0,job['absolute_deadline_unix']-time.time()))
    if absolute_deadline:
        deadline=min(deadline,start+max(0,absolute_deadline-time.time()))
    old=s.WORKSPACE/base['feature_source']
    original_folder=PROJECT/'runs/label-noise/cifar10/noise-1500'
    partial_folder=PROJECT/'runs/one-hour-bracket/cifar10/noise-1500'
    RUN.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'status':'starting','planned_models':len(job['widths']),'completed_models':0}
    def progress(**fields):
        state.update(fields);state['updated_unix']=time.time()
        s.write_json(RUN/'progress.json',state)
        if time.monotonic()>deadline:raise TimeBudgetReached('38-minute root-sweep limit reached; epoch checkpoints retained.')
    cfg={**base,'datasets':['cifar10'],'seed':job['seed'],'repeat_seeds':[],
                'widths':job['widths'],'noise_rates':[job['noise_fraction']],
         'root_epochs':job['root_epochs'],'mlp_epochs':job['mlp_epochs'],
         'depth':0,'protocol':job['policy']}
    original_fit=s.fit_node
    fit_globals=original_fit.__globals__
    original_checkpoint_writer=fit_globals['save_checkpoint']
    original_hashes={}
    try:
        with s.study_lock(RUN/'runner.lock'):
            s.fixed_definition(RUN/'definition.json',{'job':job,'base':base,'config':cfg})
            s.write_json(RUN/'last-invocation-budget.json',{'started_unix':time.time(),
                'deadline_unix':time.time()+deadline-time.monotonic(),
                'checkpoint_interval_epochs':checkpoint_interval,
                'training_calculations_and_hyperparameters_unchanged':True})
            fit_globals['save_checkpoint']=periodic_checkpoint_writer(original_checkpoint_writer,job['root_epochs'],checkpoint_interval)
            progress(status='loading_frozen_features')
            source,metadata=s.source_bundle('cifar10',base)
            s.write_json(OUTPUT/'feature-extractor.json',metadata)
            folder=RUN/'cifar10/noise-1500'
            bundle=s.noisy_bundle(source,.15,base['noise_seed'],folder)
            assert s.read_json(original_folder/'noise-manifest.json')==s.read_json(folder/'noise-manifest.json')
            s.fixed_definition(folder/'repeat-plan.json',{'widths':job['widths'],'seeds':[],
                'predeclared':True,'uses_training_only':True})
            for width in job['widths']:
                for c in range(10):
                    rel=Path(f'seed-{job["seed"]}/width-{width}/bases/class-{c}/root.pt')
                    path=original_folder/rel
                    original_hashes[str(path)]=s.file_hash(path)
            s.write_json(RUN/'preserved-source-roots.json',original_hashes)

            def continued_fit(model, images, indices, labels, path, local, identity, callback=None):
                relative=Path(path).resolve().relative_to(folder.resolve())
                partial=partial_folder/relative
                expected=s.digest({'identity':identity,'config':local,'task':s.tensor_hash(indices,labels)})
                if not Path(path).exists() and partial.exists():
                    saved=s.load_checkpoint(partial)
                    if saved['signature']==expected:
                        s.save_checkpoint(path,saved)
                        s.write_json(Path(path).with_suffix('.continuation.json'),{'mode':'continued-matching-partial-root',
                            'source':str(partial),'source_sha256':s.file_hash(partial),'starting_epoch':saved['epoch']})
                audit=followup.fork_checkpoint(original_folder/relative,path,indices,labels,local,identity,base)
                if not Path(path).with_suffix('.continuation.json').exists():s.write_json(Path(path).with_suffix('.continuation.json'),audit)
                def sparse_progress(**fields):
                    if fields.get('epoch',0)%checkpoint_interval==0 or fields.get('epoch')==job['root_epochs']:
                        if callback:callback(**fields)
                return original_fit(model,images,indices,labels,path,local,identity,sparse_progress)

            s.fit_node=continued_fit
            for width in job['widths']:
                progress(status='training_roots_only',width=width,seed=job['seed'],noise_fraction=.15)
                s.fit_stage(folder,bundle,cfg,job['seed'],width,progress,torch.device('cuda'),metadata['cnn_parameters'])
                state['completed_models']=len(list(folder.glob('seed-*/width-*/metrics.json')))
                progress(completed_models=state['completed_models'])
                s.export_results(RUN/'cifar10',OUTPUT/'cifar10',cfg)
            progress(status='final_test_evaluation')
            s.finalize_test(folder,bundle,cfg,torch.device('cuda'))
            s.export_results(RUN/'cifar10',OUTPUT/'cifar10',cfg)
            assert all(s.file_hash(Path(path))==v for path,v in original_hashes.items())
            progress(status='complete',previous_roots_verified_unchanged=True,
                elapsed_seconds=time.monotonic()-start)
    except TimeBudgetReached as exc:
        state.update(status='stopped_at_user_time_limit',reason=str(exc),elapsed_seconds=time.monotonic()-start,
            incomplete_runs_excluded=True,updated_unix=time.time())
        s.write_json(RUN/'progress.json',state);print(str(exc),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),updated_unix=time.time());s.write_json(RUN/'progress.json',state);raise
    finally:
        s.fit_node=original_fit
        fit_globals['save_checkpoint']=original_checkpoint_writer


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=PROJECT/'root-gap-config.json')
    p.add_argument('--run-dir',type=Path,default=RUN);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--checkpoint-interval',type=int,default=1)
    p.add_argument('--absolute-deadline',type=float)
    a=p.parse_args()
    if a.checkpoint_interval<1:p.error('Checkpoint interval must be positive.')
    main(a.config,a.run_dir,a.output,a.checkpoint_interval,a.absolute_deadline)
