"""Augmentation, capacity, ensembles, sample/epoch bias-variance and loss slices."""
import argparse
from contextlib import contextmanager
import gc
import os
import platform
import time
from common import *
from data import prepare,training_rows,stratified_bootstrap,test_features
from train import Trainer,BudgetExpired,evaluate,restore_forest
from analysis import summarize,landscape,majority_vote,disagreement


def job_key(job):return digest(job)[:18]


def group_key(job):
    return digest({k:v for k,v in job.items() if k not in ['epochs','depth']})[:18]


def model_folder(run,job):return run/'models'/group_key(job)/f'epochs-{job["epochs"]}'/f'depth-{job["depth"]}'


def make_plan(config):
    jobs=[]
    def add(dataset,arm='augmented',noise=0.,width=None,depth=None,epochs=None,sampler='full',fraction=1.,replicate=0,seed=None):
        job={'dataset':dataset,'augmentation':arm,'noise':noise,
            'width':config['depth_width'] if width is None else width,
            'depth':config['capacity_depth'] if depth is None else depth,
            'epochs':config['capacity_epochs'] if epochs is None else epochs,
            'sampler':sampler,'sample_fraction':fraction,'replicate':replicate,
            'seed':config['seed']+replicate if seed is None else seed}
        if job not in jobs:jobs.append(job)
    # Early coverage: real clean-label models and one landscape pair on both datasets.
    for dataset in config['datasets']:
        for arm in ['none','repeat','augmented']:add(dataset,arm)
    # Full-sample bootstrap members, with common epoch snapshots. These members
    # serve both bias-variance and 1/3/5-member majority voting without retraining.
    for epochs in config['epochs']:
        for replicate in range(config['ensemble_members']):
            for dataset in config['datasets']:
                for arm in ['none','augmented']:
                    add(dataset,arm,width=config['variance_width'],depth=1,epochs=epochs,sampler='bootstrap',replicate=replicate)
    # Fixed-root depth is varied separately from child width.
    for depth in config['depths']:
        for dataset in config['datasets']:
            for arm in ['none','augmented']:add(dataset,arm,depth=depth)
    for width in config['widths']:
        for dataset in config['datasets']:
            for arm in ['none','augmented']:add(dataset,arm,width=width,depth=config['capacity_depth'])
    for fraction in config['sample_fractions']:
        if fraction==1:continue
        for epochs in config['epochs']:
            for replicate in range(config['variance_replicates']):
                for dataset in config['datasets']:
                    for arm in ['none','augmented']:
                        add(dataset,arm,width=config['variance_width'],depth=1,epochs=epochs,sampler='bootstrap',fraction=fraction,replicate=replicate)
    for rate in config['noise_rates']:
        if rate==0:continue
        for width in config.get('noise_widths',config['widths']):
            for dataset in config['datasets']:
                for arm in ['none','augmented']:add(dataset,arm,noise=rate,width=width,depth=config['capacity_depth'])
    def priority(job):
        if job['sampler']=='full' and not job['noise'] and job['width']==config['depth_width'] and job['depth']==config['capacity_depth']:
            return 0
        if job['sampler']=='bootstrap' and job['epochs']==min(config['epochs']):return 1
        if job['sampler']=='full' and not job['noise'] and job['width']==max(config['widths']) and job['depth']==config['capacity_depth']:return 2
        if job['sampler']=='full' and not job['noise'] and job['depth']>config['capacity_depth']:return 3
        if job['sampler']=='bootstrap' and job['sample_fraction']==1:return 4
        if job['sampler']=='full' and not job['noise']:return 5
        if job['sampler']=='bootstrap':return 6
        return 7
    return sorted(jobs,key=priority)


def labels_for_job(bundle,job,config):
    labels=bundle['splits']['train']['y'].clone()
    if not job['noise']:return labels
    if job['noise']!=.15:raise ValueError('Only the recorded 15% control is implemented.')
    path=WORKSPACE/config['reuse_noise_map']/job['dataset']/'noise-1500/noise-manifest.json'
    mapping={c['id']:c['training_label'] for c in read_json(path)['changed_samples']}
    for i,id_ in enumerate(bundle['splits']['train']['ids']):
        if id_ in mapping:labels[i]=mapping[id_]
    return labels


def fit_job(run,output,bundle,job,config,device,progress,deadline):
    destination=model_folder(run,job)
    if (destination/'metrics.json').exists():return read_json(destination/'metrics.json')
    progress(stage='training',job=job,job_id=job_key(job))
    clean=bundle['splits']['train']['y']
    draws=stratified_bootstrap(clean,job['sample_fraction'],800000+job['seed']) if job['sampler']=='bootstrap' else torch.arange(len(clean))
    fingerprint=digest({'features':bundle['signature'],'augmentation':job['augmentation'],
        'draws':tensor_hash(draws),'labels':tensor_hash(labels_for_job(bundle,job,config))})
    x,_=training_rows(bundle,draws,job['augmentation'])
    views=config['views'] if job['augmentation'] in ['repeat','augmented'] else 1
    y=labels_for_job(bundle,job,config)[draws].repeat(views)
    unique=draws.unique(sorted=True)
    epoch_folder=destination.parent
    earlier_epochs=sorted([p for p in epoch_folder.parent.glob('epochs-*') if int(p.name.split('-')[1])<job['epochs']],key=lambda p:int(p.name.split('-')[1]))
    earlier_depths=sorted([p for p in epoch_folder.glob('depth-*/model.pt') if int(p.parent.name.split('-')[1])<job['depth']],key=lambda p:int(p.parent.name.split('-')[1]))
    previous=earlier_epochs[-1] if earlier_epochs else None
    existing=restore_forest(earlier_depths[-1])[0] if earlier_depths else None
    node_config={**config,'epochs':job['epochs']}
    trainer=Trainer(epoch_folder,x.to(device),y.to(device),bundle['classes'],node_config,
        fingerprint,job['seed'],job['width'],job['depth'],progress,deadline,previous,existing)
    started=time.monotonic();forest=trainer.fit()
    record={'dataset':job['dataset'],'job':job,'job_id':job_key(job),'classes':bundle['classes'],
        'debug_subset':bundle['debug_subset'],'sample_draws':len(draws),'unique_originals':len(unique),
        'augmentation_views_per_original':views,'finite_training_views':len(y),
        'feature_signature':bundle['signature'],'size':capacity(forest,bundle['cnn_parameters']),
        'seconds_this_invocation':time.monotonic()-started,'model_path':(destination/'model.pt').relative_to(run).as_posix(),
        'predictions_path':(destination/'predictions.pt').relative_to(run).as_posix(),
        'test_status':'withheld_until_primary_and_training_selected_repeat_plans_complete',
        'bias_variance_scope':config['bias_variance_scope']}
    record['train_views'],_,_=evaluate(forest,x,y,config['eval_batch'])
    record['train_selected_unique'],_,_=evaluate(forest,bundle['splits']['train']['x'][unique],clean[unique],config['eval_batch'])
    record['train_selected_unique_label_reference']='Original clean labels; noisy fitting is reported separately.'
    study_labels=labels_for_job(bundle,job,config)
    record['changed_training_draws']=int((study_labels[draws]!=clean[draws]).sum())
    record['actual_noise_fraction_in_draws']=record['changed_training_draws']/len(draws)
    record['train_canonical_study_labels'],_,_=evaluate(forest,bundle['splits']['train']['x'][unique],labels_for_job(bundle,job,config)[unique],config['eval_batch'])
    record['node_optimizer_updates']=sum(((t['examples']+config['batch']-1)//config['batch'])*job['epochs']
        for t in trainer.tasks.values() if not t['base'].get('frozen_original_cnn_root'))
    val=bundle['splits']['validation']
    record['validation'],vp,vl=evaluate(forest,val['x'],val['y'],config['eval_batch'])
    record['convergence_flags']=[{'node':key,**t['base']} for key,t in trainer.tasks.items() if t['base'].get('loss_still_falling')]
    save_checkpoint(destination/'model.pt',{'spec':forest.spec(),'model':cpu_state(forest),'job':job,'feature_signature':bundle['signature'],'tasks':trainer.tasks})
    save_checkpoint(destination/'sample-plan.pt',{'draws':draws,'unique':unique,'fingerprint':fingerprint})
    save_checkpoint(destination/'predictions.pt',{'validation_probabilities':vp,'validation_prediction':vl,'validation_y':val['y']})
    write_json(destination/'metrics.json',record)
    if job['sampler']=='full' and job['noise']==0 and job['width']==config['depth_width'] and job['depth']==config['capacity_depth']:
        from deploy import export_model
        export_model(bundle,[load_checkpoint(destination/'model.pt')],output/'models'/f"{job['dataset']}-{job['augmentation']}-single.pt",
            {'job':job,'debug_subset':bundle['debug_subset'],'selection':'Predeclared reference configuration; no held-out selection.'})
    print(f"{job['dataset']} {job['augmentation']} noise={job['noise']:.0%} n={len(draws)} w={job['width']} d={job['depth']} E={job['epochs']} seed={job['seed']}: views train {record['train_views']['accuracy']:.3%}, validation {record['validation']['accuracy']:.3%}",flush=True)
    del trainer,forest,x,existing
    if device.type=='cuda':torch.cuda.empty_cache()
    return record


def repeats_from_training(primary,run,config):
    repeats=[]
    for dataset in config['datasets']:
        for arm in ['none','augmented']:
            for noise in config['noise_rates']:
                group=[j for j in primary if j['dataset']==dataset and j['augmentation']==arm and j['noise']==noise
                    and j['sampler']=='full' and j['depth']==config['capacity_depth'] and j['width'] in config['widths']]
                rows=sorted(group,key=lambda j:j['width'])
                first=next((i for i,j in enumerate(rows) if read_json(model_folder(run,j)/'metrics.json')['train_views']['errors']==0),None)
                positions=([max(0,first-1),first,min(len(rows)-1,first+1)] if first is not None else [0,len(rows)//2,len(rows)-1])
                for i in sorted(set(positions)):
                    for seed in config['repeat_seeds']:repeats.append({**rows[i],'seed':seed})
    return repeats


def evaluate_reference(run,output,bundles,config,primary,device):
    """One fixed comparison can finish before the wider overnight study."""
    if config.get('debug_train_limit') is not None:return False
    refs=[j for j in primary if j['sampler']=='full' and j['noise']==0 and
          j['width']==config['depth_width'] and j['depth']==config['capacity_depth'] and
          j['epochs']==config['capacity_epochs'] and j['seed']==config['seed']]
    assert len(refs)==3*len(config['datasets'])
    if not all((model_folder(run,j)/'metrics.json').exists() for j in refs):return False
    gate=run/'reference-test-plan.json'
    fixed_json(gate,{'jobs':refs,'policy':'All fixed reference models complete; no held-out selection; all later model settings already declared.'})
    rows=[]
    for dataset in config['datasets']:
        jobs=[j for j in refs if j['dataset']==dataset]
        pending=[j for j in jobs if 'test' not in read_json(model_folder(run,j)/'metrics.json')]
        heldout=test_features(dataset,bundles[dataset],config['feature_source']) if pending else None
        for job in jobs:
            path=model_folder(run,job);record=read_json(path/'metrics.json')
            if 'test' not in record:
                forest,_=restore_forest(path/'model.pt',device)
                record['test'],p,pred=evaluate(forest,heldout['x'],heldout['y'],config['eval_batch'])
                record['test_status']='evaluated_after_all_fixed_reference_models_complete; reused_test_exploratory'
                saved=load_checkpoint(path/'predictions.pt');saved.update(test_probabilities=p,test_prediction=pred,test_y=heldout['y'])
                save_checkpoint(path/'predictions.pt',saved);write_json(path/'metrics.json',record)
                del forest
            rows.append(record)
    write_json(output/'reference-test.json',{'complete':True,'fixed_models':len(refs),'records':rows})
    return True


@contextmanager
def keep_awake():
    result=False
    if os.name=='nt':
        import ctypes
        result=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:yield result
    finally:
        if result:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def finalize(run,output,bundles,config,primary,repeats,device):
    all_jobs=primary+repeats
    assert all((model_folder(run,j)/'metrics.json').exists() for j in all_jobs)
    if config.get('debug_train_limit') is not None:return
    for dataset in config['datasets']:
        heldout=test_features(dataset,bundles[dataset],config['feature_source'])
        for job in [j for j in all_jobs if j['dataset']==dataset]:
            path=model_folder(run,job);record=read_json(path/'metrics.json')
            if 'test' in record:continue
            forest,_=restore_forest(path/'model.pt',device)
            record['test'],p,pred=evaluate(forest,heldout['x'],heldout['y'],config['eval_batch'])
            record['test_status']='evaluated_after_fixed_primary_and_training_selected_repeats; reused_test_exploratory'
            saved=load_checkpoint(path/'predictions.pt');saved.update(test_probabilities=p,test_prediction=pred,test_y=heldout['y'])
            save_checkpoint(path/'predictions.pt',saved);write_json(path/'metrics.json',record)
            del forest
        # Majority-vote test evaluation at every completed epoch snapshot.
        for epochs in config['epochs']:
            for arm in ['none','augmented']:
                members=[j for j in primary if j['dataset']==dataset and j['augmentation']==arm and j['sampler']=='bootstrap'
                    and j['sample_fraction']==1 and j['epochs']==epochs]
                members.sort(key=lambda j:j['replicate'])
                arrays=[load_checkpoint(model_folder(run,j)/'predictions.pt') for j in members]
                rows=[]
                for count in [1,3,5]:
                    if len(arrays)<count:continue
                    predictions=torch.stack([a['test_prediction'] for a in arrays[:count]])
                    prediction,_=majority_vote(predictions,bundles[dataset]['classes'])
                    rows.append({'members':count,'test_accuracy':float((prediction==heldout['y']).float().mean()),
                        'pairwise_disagreement':disagreement(predictions),'epochs_per_child':epochs,'augmentation':arm})
                write_json(output/f'{dataset}-{arm}-E{epochs}-ensemble-test.json',rows)


def run(config_path,run_dir,output,hours,device):
    config=read_json(config_path);run_dir.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    primary=make_plan(config);started=time.monotonic();deadline=started+hours*3600
    state={'pid':os.getpid(),'status':'starting','planned_primary_models':len(primary),'completed_primary_models':0,'started_unix':time.time(),
           'wall_clock_limit_hours':hours,'device':str(device),'debug_subset':config.get('debug_train_limit') is not None}
    def progress(**fields):
        state.update(fields);state['updated_unix']=time.time();write_json(run_dir/'progress.json',state)
        if time.monotonic()>=deadline:raise BudgetExpired()
    with study_lock(run_dir/'runner.lock'),keep_awake() as wake:
        state['temporary_system_awake_request']=wake
        definition={'config':config,'source_hashes':{p.name:file_hash(p) for p in PROJECT.glob('*.py')}}
        fixed_json(run_dir/'definition.json',definition)
        fixed_json(run_dir/'primary-plan.json',primary)
        write_json(run_dir/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'started_unix':time.time()})
        bundles={}
        try:
            for dataset in config['datasets']:
                progress(status='running',stage='preparing',dataset=dataset)
                bundles[dataset]=prepare(dataset,run_dir/'features'/dataset,config,device,progress)
            for job in primary:
                record=fit_job(run_dir,output,bundles[job['dataset']],job,config,device,progress,deadline)
                state['completed_primary_models']=sum((model_folder(run_dir,j)/'metrics.json').exists() for j in primary)
                progress(stage='completed_model',last_completed_job=job,completed_primary_models=state['completed_primary_models'])
                summarize(run_dir,output)
                if job['sampler']=='full' and job['noise']==0 and job['width']==config['depth_width']:
                    evaluate_reference(run_dir,output,bundles,config,primary,device)
                if job['sampler']=='bootstrap' and job['sample_fraction']==1 and job['epochs']==max(config['epochs']):
                    from deploy import export_ensembles
                    export_ensembles(run_dir,output,bundles,config,primary)
                if state['completed_primary_models']%10==0:
                    from plots import plot_all
                    plot_all(run_dir,output)
                # Representative fixed-model landscapes, selected by plan,
                # not by their validation score; early output covers the brief.
                if job['sampler']=='full' and job['noise']==0 and job['width']==config['depth_width'] and job['depth']==config['capacity_depth']:
                    lf=output/'landscapes'/f"{job['dataset']}-{job['augmentation']}"
                    if not (lf/'landscape.json').exists():
                        forest,_=restore_forest(model_folder(run_dir,job)/'model.pt',device)
                        x,y=training_rows(bundles[job['dataset']],torch.arange(len(bundles[job['dataset']]['splits']['train']['y'])),job['augmentation'])
                        val=bundles[job['dataset']]['splits']['validation']
                        landscape(forest,x,y,val['x'],val['y'],lf,config['landscape'],deadline,progress)
                        del forest,x
            repeats=repeats_from_training(primary,run_dir,config)
            fixed_json(run_dir/'repeat-plan.json',{'selection':'training-view interpolation only; below/near/above, or low/middle/high if none interpolates','jobs':repeats})
            for job in repeats:
                fit_job(run_dir,output,bundles[job['dataset']],job,config,device,progress,deadline)
                summarize(run_dir,output)
            progress(stage='final_test_evaluation')
            finalize(run_dir,output,bundles,config,primary,repeats,device)
            summarize(run_dir,output)
            from plots import plot_all
            plot_all(run_dir,output)
            from deploy import export_ensembles
            export_ensembles(run_dir,output,bundles,config,primary)
            progress(status='complete',stage='complete',elapsed_seconds=time.monotonic()-started,
                     completed_repeat_models=len(repeats),test_evaluated=config.get('debug_train_limit') is None)
        except (BudgetExpired,TimeoutError):
            state.update(status='stopped_at_time_limit',stage='checkpoints_saved_resume_available',updated_unix=time.time(),elapsed_seconds=time.monotonic()-started)
            write_json(run_dir/'progress.json',state);summarize(run_dir,output)
            from plots import plot_all
            plot_all(run_dir,output)
            print('Time limit reached. Completed measurements retained; incomplete jobs excluded. Resume with the same command.',flush=True)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),updated_unix=time.time())
            write_json(run_dir/'progress.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config.json')
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'runs/augmentation-study')
    parser.add_argument('--output',type=Path,default=PROJECT/'results/augmentation-study')
    parser.add_argument('--hours',type=float,default=10)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=parser.parse_args()
    chosen='cuda' if torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
    if a.device!='auto':chosen=a.device
    if a.hours<=0:parser.error('hours must be positive')
    run(a.config,a.run_dir,a.output,a.hours,torch.device(chosen))
