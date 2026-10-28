"""Frozen, label-free random convolutional features for neural correction trees."""
from pathlib import Path
import argparse
import csv
import gc
import os
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'runs/matplotlib-cache'))
sys.path.insert(0,str(WORKSPACE/'neural_tree_v3'))
import numpy as np
import torch
from torch import nn
import study as shared
from width32_followup import refit, saturated
from v2_tree.common import (cpu_state, digest, file_hash, load_checkpoint, read_json,
    save_checkpoint, seed_everything, study_lock, tensor_hash, write_json)
from v2_tree.data import make_cache, normalized
from v2_tree.models import correction_targets
from v2_tree.train import evaluate, fit_node, score_metrics, scores_on
from v2_tree.study import fixed_definition, protected_originals, verify_originals


class RandomCNN(nn.Module):
    def __init__(self, channels=(16,32,64), grid=2):
        super().__init__()
        self.channels, self.grid = tuple(channels), grid
        blocks, incoming = [], 3
        for outgoing in channels:
            blocks += [nn.Conv2d(incoming,outgoing,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]
            incoming=outgoing
        blocks += [nn.AdaptiveAvgPool2d((grid,grid)),nn.Flatten()]
        self.layers=nn.Sequential(*blocks)
        for parameter in self.parameters(): parameter.requires_grad_(False)

    @property
    def dimension(self): return self.channels[-1]*self.grid**2

    def forward(self,x): return self.layers(x)


class RandomForest(shared.FeatureForest):
    def size(self):
        size=super().size()
        size['correction_nodes']=size['nodes']-len(self.roots)
        reachable=[]
        def visit(node):
            reachable.append(node)
            if node.A is not None and float(node.alpha.detach())>0: visit(node.A)
            if node.B is not None and float(node.beta.detach())>0: visit(node.B)
        for root in self.roots: visit(root)
        size['active_nodes']=len(reachable)
        size['active_parameters']=sum(sum(p.numel() for p in node.base.parameters())+3 for node in reachable)
        size['frozen_root_parameters']=sum(sum(p.numel() for p in root.base.parameters()) for root in self.roots)
        return size


class ImageClassifier(nn.Module):
    def __init__(self, cnn, forest, image_mean, image_std, feature_mean, feature_std):
        super().__init__(); self.cnn,self.forest=cnn,forest
        for parameter in cnn.parameters(): parameter.requires_grad_(False)
        for root in forest.roots:
            for parameter in root.base.parameters(): parameter.requires_grad_(False)
        for name,value in [('image_mean',image_mean),('image_std',image_std),
                           ('feature_mean',feature_mean),('feature_std',feature_std)]:
            self.register_buffer(name,value.clone())

    def forward(self,image):
        pixels=(image-self.image_mean[None,:,None,None])/self.image_std[None,:,None,None]
        features=(self.cnn(pixels)-self.feature_mean)/self.feature_std
        return self.forest(features)


def load_deployment(path,device='cpu'):
    saved=load_checkpoint(path)
    cnn=RandomCNN(saved['channels'],saved['grid']); cnn.load_state_dict(saved['cnn_model'])
    forest=RandomForest.from_spec(saved['forest_spec']); forest.load_state_dict(saved['forest_model'])
    model=ImageClassifier(cnn,forest,saved['image_mean'],saved['image_std'],saved['feature_mean'],saved['feature_std'])
    return model.to(device).eval(),saved


def random_extractor(folder,config,device):
    path=Path(folder)/'random-cnn.pt'
    identity={'seed':config['feature_seed'],'channels':config['channels'],'grid':config['spatial_grid']}
    seed_everything(config['feature_seed'])
    cnn=RandomCNN(config['channels'],config['spatial_grid']).to(device).eval()
    if path.exists():
        saved=load_checkpoint(path)
        if saved['identity']!=identity: raise ValueError('Random feature identity changed')
        cnn.load_state_dict(saved['model'])
    else:
        save_checkpoint(path,{'identity':identity,'model':cpu_state(cnn),'supervised_updates':0,
            'trained_epochs':0,'classification_head':False})
    return cnn,file_hash(path)


def feature_bundle(cnn,cache,folder,cnn_hash,config,device):
    path=Path(folder)/'features.pt'
    signature=digest({'pixels':cache['fingerprint'],'random_cnn':cnn_hash,
        'test':tensor_hash(cache['splits']['test']['pixels'],cache['splits']['test']['y'])})
    if path.exists():
        bundle=load_checkpoint(path)
        if bundle['signature']!=signature: raise ValueError('Feature cache identity changed')
        return bundle
    bundle={'signature':signature,'classes':len(cache['classes']),'dimension':cnn.dimension,'splits':{}}
    cnn.eval()
    with torch.no_grad():
        for split,data in cache['splits'].items():
            z=torch.cat([cnn(normalized(batch,cache,device)).cpu() for batch in data['pixels'].split(config['feature_batch'])])
            bundle['splits'][split]={'x':z,'y':data['y'],'ids':data['ids']}
    z=bundle['splits']['train']['x']
    bundle['feature_mean'],bundle['feature_std']=z.mean(0),z.std(0,unbiased=False).clamp_min(1e-6)
    for data in bundle['splits'].values(): data['x']=(data['x']-bundle['feature_mean'])/bundle['feature_std']
    _,inverse=np.unique(bundle['splits']['train']['x'].numpy(),axis=0,return_inverse=True)
    grouped=np.zeros((int(inverse.max())+1,bundle['classes']),dtype=np.int64)
    np.add.at(grouped,(inverse,bundle['splits']['train']['y'].numpy()),1)
    unavoidable=int((grouped.sum(1)-grouped.max(1)).sum())
    bundle['diagnostics']={'unique_training_features':len(grouped),'minimum_errors_from_feature_collisions':unavoidable,
        'training_accuracy_upper_bound':1-unavoidable/len(z),'cnn_supervised_updates':0}
    save_checkpoint(path,bundle)
    return bundle


class Trainer:
    def __init__(self,folder,config,x,signature,class_index,seed,width,progress,existing_paths=()):
        self.folder,self.config,self.x=Path(folder),config,x
        self.signature,self.class_index,self.seed,self.width=signature,class_index,seed,width
        self.progress,self.tasks,self.used=progress,{},0
        self.unvisited_existing=set(existing_paths)

    def grow(self,indices,y,depth_limit,cap,path='root',depth=0):
        self.unvisited_existing.discard(path)
        self.used+=1
        seed_everything(self.seed+self.class_index+int(digest(path)[:7],16))
        width=self.config['root_width'] if depth==0 else self.width
        node=shared.FeatureNode(self.x.shape[1],width).to(self.x.device)
        identity={'path':path,'depth':depth,'kind':'mlp','class':self.class_index,'features':self.signature}
        base_config={key:self.config[key] for key in ('root_epochs','mlp_epochs','mlp_batch','lr','eval_batch')}
        base_config.update(width=width,seed=self.seed)
        base_path=(self.folder/'roots'/f'seed-{self.seed}'/f'class-{self.class_index}.pt' if depth==0 else
                   self.folder/'bases'/f'seed-{self.seed}'/f'width-{self.width}'/f'class-{self.class_index}'/(path+'.pt'))
        stats=fit_node(node.base,self.x,indices,y,base_path,base_config,identity,
            lambda **fields:self.progress(class_index=self.class_index,**fields))
        if depth==0:
            for parameter in node.base.parameters(): parameter.requires_grad_(False)
        base_scores=scores_on(node.base,self.x,indices,self.config['eval_batch'])
        before=score_metrics(base_scores,y)
        self.tasks[path]={'base':stats,'examples':len(y),'errors_before':before[0]}
        if depth<depth_limit and before[0]:
            for side,(targets,mask) in correction_targets(base_scores>=0,y).items():
                child_path=path+'.'+side
                has_room=child_path in self.unvisited_existing or self.used+len(self.unvisited_existing)<cap
                if has_room and targets[mask].sum()>0:
                    setattr(node,side,self.grow(indices[mask],targets[mask],depth_limit,cap,path+'.'+side,depth+1))
        if node.A is not None or node.B is not None:
            a=(scores_on(node.A,self.x,indices,self.config['eval_batch'])>=0).float() if node.A is not None else torch.zeros_like(y)
            b=(scores_on(node.B,self.x,indices,self.config['eval_batch'])>=0).float() if node.B is not None else torch.zeros_like(y)
            weights,correction=refit(torch.stack([base_scores,a,b],1),y,
                self.folder/'corrections'/f'seed-{self.seed}'/f'width-{self.width}'/f'depth-{depth_limit}'/f'class-{self.class_index}'/(path+'.pt'),
                self.config['correction_epochs'],identity,'classification_refit')
            with torch.no_grad():
                node.alpha.copy_(weights[0]); node.beta.copy_(weights[1]); node.offset.copy_(weights[2])
            if score_metrics(scores_on(node,self.x,indices,self.config['eval_batch']),y)>before:
                with torch.no_grad(): node.alpha.zero_(); node.beta.zero_(); node.offset.zero_()
            self.tasks[path]['correction']=correction
        self.tasks[path]['errors_after']=score_metrics(scores_on(node,self.x,indices,self.config['eval_batch']),y)[0]
        return node


def fit_stage(folder,bundle,config,seed,width,depth,experiment,device,progress):
    destination=Path(folder)/'stages'/f'seed-{seed}'/f'width-{width}'/f'depth-{depth}'
    if (destination/'metrics.json').exists(): return read_json(destination/'metrics.json')
    classes=bundle['classes']; count=1 if classes==2 else classes
    features=bundle['dimension']; root_parameters=(features+2)*config['root_width']+4
    child_parameters=(features+2)*width+4
    cap=min(config['max_nodes_per_tree'],max(1,1+(config['max_tree_parameters']//count-root_parameters)//child_parameters))
    previous=sorted([p for p in destination.parent.glob('depth-*/metrics.json') if int(p.parent.name.split('-')[1])<depth],
        key=lambda p:int(p.parent.name.split('-')[1]))
    saved=None
    if previous:
        earlier=read_json(previous[-1]); saved=load_checkpoint(previous[-1].parent/'forest.pt')
        if saturated(saved,cap):
            record={**earlier,'depth_limit':depth,'experiment':experiment,'reused_saturated_model':True,
                'saturated_from_depth':earlier.get('saturated_from_depth',earlier['depth_limit'])}
            save_checkpoint(destination/'forest.pt',saved); write_json(destination/'metrics.json',record)
            return record
    started=time.perf_counter()
    x,y=bundle['splits']['train']['x'].to(device),bundle['splits']['train']['y'].to(device)
    roots,tasks=[],{}
    for i,c in enumerate([1] if classes==2 else range(classes)):
        old_root=shared.FeatureNode.from_spec(saved['spec']['roots'][i]) if saved else None
        existing=[p for p,_,_ in old_root.walk()] if old_root else []
        trainer=Trainer(folder,config,x,bundle['signature'],c,seed,width,progress,existing)
        roots.append(trainer.grow(torch.arange(len(y),device=device),(y==c).float(),depth,cap).cpu())
        tasks[str(c)]=trainer.tasks
    forest=RandomForest(classes,roots).to(device)
    record={'seed':seed,'width':width,'depth_limit':depth,'experiment':experiment,'size':forest.size(),
        'node_cap_per_tree':cap,'feature_signature':bundle['signature'],'selection_uses_test':False,
        'test_repeated_exploratory':True,'fixed_root_width':config['root_width']}
    for split,data in bundle['splits'].items(): record[split]=evaluate(forest,data['x'].to(device),data['y'].to(device),config['eval_batch'])
    record['convergence_flags']=[{'class':c,'node':p,'last_errors':t['base']['history'][-1]['errors']}
        for c,ts in tasks.items() for p,t in ts.items() if len(t['base']['history'])>=5 and
        t['base']['history'][-1]['errors']>0 and t['base']['history'][-1]['loss']<.99*t['base']['history'][-5]['loss']]
    record['seconds']=time.perf_counter()-started
    save_checkpoint(destination/'forest.pt',{'spec':forest.spec(),'model':cpu_state(forest),'tasks':tasks})
    write_json(destination/'metrics.json',record)
    print(f"{experiment} seed {seed} width {width} depth {depth}: train {record['train']['accuracy']:.3%}, validation {record['validation']['accuracy']:.3%}, test {record['test']['accuracy']:.3%}",flush=True)
    return record


def export_results(folder,output,cnn_parameters,config):
    records=[read_json(p) for p in Path(folder).glob('stages/seed-*/width-*/depth-*/metrics.json')]
    records.sort(key=lambda r:(r['seed'],r['depth_limit'],r['width']))
    for r in records:
        r['total_parameters']=cnn_parameters+r['size']['parameters']
        r['total_active_parameters']=cnn_parameters+r['size']['active_parameters']
    write_json(Path(output)/'measurements.json',records)
    fields=['experiment','seed','width','depth_limit','total_parameters','total_active_parameters','nodes',
        'train_accuracy','validation_accuracy','test_accuracy','seconds','reused_saturated_model']
    with (Path(output)/'measurements.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({**{k:r[k] for k in fields[:6]},'nodes':r['size']['nodes'],
                **{split+'_accuracy':r[split]['accuracy'] for split in ('train','validation','test')},
                'seconds':r.get('seconds',0),'reused_saturated_model':r.get('reused_saturated_model',False)})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for experiment in ('primary','deeper'):
        primary=[r for r in records if r['seed']==config['seed'] and r['experiment']==experiment and not r.get('reused_saturated_model')]
        if not primary: continue
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        groups=sorted(set(r['depth_limit'] for r in primary))
        for depth in groups:
            rows=sorted([r for r in primary if r['depth_limit']==depth],key=lambda r:r['width'])
            for axis,split in zip(axes,('train','validation','test')):
                axis.plot([r['total_parameters'] for r in rows],[100*(1-r[split]['accuracy']) for r in rows],'o-',label=f'depth {depth}')
                zero=next((r for r in rows if r['train']['errors']==0),None)
                if zero: axis.scatter([zero['total_parameters']],[100*(1-zero[split]['accuracy'])],marker='*',s=150,color='black',zorder=5)
                axis.set(xscale='log',xlabel='Total deployed parameters (random CNN counted once)',ylabel=f'{split.capitalize()} error (%)')
                axis.grid(alpha=.2); axis.legend()
        fig.suptitle(f'Frozen random CNN: {experiment} correction study; test curves exploratory')
        fig.tight_layout(); fig.savefig(Path(output)/(experiment+'-parameter-error.png'),dpi=170); plt.close(fig)
    summaries=[]
    for experiment in ('primary','deeper'):
        settings=sorted(set((r['width'],r['depth_limit']) for r in records if r['experiment']==experiment))
        for width,depth in settings:
            rows=[r for r in records if r['experiment']==experiment and r['width']==width and r['depth_limit']==depth]
            if len(rows)<2: continue
            row={'experiment':experiment,'width':width,'depth':depth,'seeds':[r['seed'] for r in rows],
                'parameters_min':min(r['total_parameters'] for r in rows),'parameters_max':max(r['total_parameters'] for r in rows)}
            for split in ('train','validation','test'):
                errors=np.asarray([1-r[split]['accuracy'] for r in rows])
                row[split]={'mean_error':float(errors.mean()),'standard_error':float(errors.std(ddof=1)/np.sqrt(len(errors)))}
            summaries.append(row)
    write_json(Path(output)/'seed-summary.json',{'conditional_on_one_random_extractor':True,'records':summaries})
    return records


def representative_configs(records,config,experiment):
    rows=[r for r in records if r['seed']==config['seed'] and r['experiment']==experiment]
    rows.sort(key=lambda r:r['total_parameters'])
    if not rows: return []
    crossing=next((i for i,r in enumerate(rows) if r['train']['errors']==0),None)
    positions=sorted(set([max(0,crossing-1),crossing,min(len(rows)-1,crossing+1)])) if crossing is not None else sorted(set([0,len(rows)//2,len(rows)-1]))
    return [{'width':rows[i]['width'],'depth':rows[i]['depth_limit']} for i in positions]


def run(config_path,run_dir,output,device,limit=None):
    config=read_json(config_path); run_dir,output=Path(run_dir).resolve(),Path(output).resolve()
    if any(p==WORKSPACE/name or WORKSPACE/name in p.parents for p in (run_dir,output) for name in ('neural_tree','neural_tree_v2','neural_tree_v3')):
        raise ValueError('Random-CNN outputs must be separate from all previous studies')
    status={'status':'preparing','pid':os.getpid(),'updated_unix':time.time()}; last_write=[0.]
    def progress(**fields):
        status.update(fields); status['updated_unix']=time.time()
        if 'epoch' in fields and time.time()-last_write[0]<5: return
        try: write_json(run_dir/'progress.json',status); last_write[0]=time.time()
        except PermissionError: pass
    with study_lock(run_dir/'runner.lock'):
        code=[Path(__file__),WORKSPACE/'neural_tree_v3/study.py',WORKSPACE/'neural_tree_v3/width32_followup.py',WORKSPACE/'neural_tree/ell409_tree/models.py']+list((WORKSPACE/'neural_tree_v2/v2_tree').glob('*.py'))
        fixed_definition(run_dir/'definition.json',{'config':config,'limit':limit,'code':{str(p):file_hash(p) for p in code},
            'manifests':{ds:file_hash(WORKSPACE/'neural_tree/data'/ds/'manifest.json') for ds in config['datasets']}})
        if not (run_dir/'preserved-originals.json').exists():
            snapshot=protected_originals(WORKSPACE/'neural_tree')
            for name in ('neural_tree_v2','neural_tree_v3'):
                results=WORKSPACE/name/'results'
                # Debug runs may overlap the earlier follow-up; full runs are queued after it.
                snapshot.update({str(p):file_hash(p) for p in results.rglob('*') if p.is_file() and
                    not (limit is not None and 'width32-followup' in p.parts)})
            write_json(run_dir/'preserved-originals.json',snapshot)
        snapshot=read_json(run_dir/'preserved-originals.json'); verify_originals(snapshot)
        write_json(run_dir/'environment.json',{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),
            'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None})
        try:
            for dataset in config['datasets']:
                folder,destination=run_dir/dataset,output/dataset
                progress(status='extracting_random_features',dataset=dataset)
                manifest=WORKSPACE/'neural_tree/data'/dataset/'manifest.json'
                source=WORKSPACE/'neural_tree_v3/runs/cnn80-depth'/dataset/'pixels.pt'
                if limit is None and source.exists():
                    cache=load_checkpoint(source)
                    expected=digest({'manifest_file':file_hash(manifest),'resolution':config['resolution'],'limit':None})
                    if cache['signature']!=expected: raise ValueError('Reused pixels do not match splits/resolution')
                else: cache=make_cache(manifest,folder/'pixels.pt',config['resolution'],limit)
                cnn,cnn_hash=random_extractor(folder,config,device)
                cache['splits']['test']=shared.test_pixels(manifest,config['resolution'],limit)
                bundle=feature_bundle(cnn,cache,folder,cnn_hash,config,device)
                cnn_parameters=sum(p.numel() for p in cnn.parameters())
                write_json(destination/'feature-extractor.json',{'type':'frozen random CNN','parameters':cnn_parameters,
                    'features':cnn.dimension,'supervised_updates':0,'feature_seed':config['feature_seed'],'sha256':cnn_hash,
                    'diagnostics':bundle['diagnostics'],'limit':limit,'full_dataset':limit is None})
                progress(status='training_fixed_roots',seed=config['seed'])
                fit_stage(folder,bundle,config,config['seed'],config['root_width'],0,'baseline',device,progress)
                for width in config['widths']:
                    progress(status='primary_sweep',width=width,depth_limit=config['primary_depth'])
                    fit_stage(folder,bundle,config,config['seed'],width,config['primary_depth'],'primary',device,progress)
                    records=export_results(folder,destination,cnn_parameters,config); progress(completed_records=len(records))
                primary=[r for r in records if r['seed']==config['seed'] and r['experiment']=='primary']
                deeper=not any(r['train']['errors']==0 for r in primary)
                write_json(folder/'fallback-decision.json',{'run_deeper':deeper,'trigger_uses_training_only':True})
                if deeper:
                    for width in config['fallback_widths']:
                        for depth in config['fallback_depths']:
                            progress(status='deeper_sweep',width=width,depth_limit=depth)
                            fit_stage(folder,bundle,config,config['seed'],width,depth,'deeper',device,progress)
                            records=export_results(folder,destination,cnn_parameters,config); progress(completed_records=len(records))
                repeats={ex:representative_configs(records,config,ex) for ex in ('primary','deeper')}
                write_json(folder/'repeat-plan.json',{'uses_training_only':True,'conditional_on_one_random_extractor':True,'configs':repeats})
                for seed in config['repeat_seeds']:
                    for experiment,jobs in repeats.items():
                        for job in jobs:
                            progress(status='repeat_training',seed=seed,width=job['width'],depth_limit=job['depth'])
                            fit_stage(folder,bundle,config,seed,job['width'],job['depth'],experiment,device,progress)
                            records=export_results(folder,destination,cnn_parameters,config); progress(completed_records=len(records))
                primary=[r for r in records if r['seed']==config['seed']]
                fitting=[r for r in primary if r['train']['accuracy']>=config['near_perfect']]
                selected=min(fitting,key=lambda r:r['total_parameters']) if fitting else max(primary,key=lambda r:(r['train']['accuracy'],-r['total_parameters']))
                saved=load_checkpoint(folder/'stages'/f"seed-{selected['seed']}"/f"width-{selected['width']}"/f"depth-{selected['depth_limit']}"/'forest.pt')
                save_checkpoint(destination/'deployment.pt',{'channels':config['channels'],'grid':config['spatial_grid'],
                    'cnn_model':cpu_state(cnn),'forest_spec':saved['spec'],'forest_model':saved['model'],
                    'image_mean':cache['mean'],'image_std':cache['std'],'feature_mean':bundle['feature_mean'],
                    'feature_std':bundle['feature_std'],'selection':selected,'cnn_hash':cnn_hash,'classes':cache['classes']})
                write_json(destination/'selection.json',{'selection':selected,'training_target_met':bool(fitting),'uses_test':False})
                write_json(destination/'capacity-observations.json',{'first_zero_error':next(({'width':r['width'],'depth':r['depth_limit'],'experiment':r['experiment']} for r in sorted(primary,key=lambda r:r['total_parameters']) if r['train']['errors']==0),None),
                    'deeper_study_triggered':deeper,'no_guaranteed_double_descent':True,'test_curves_exploratory':True})
                if file_hash(folder/'random-cnn.pt')!=cnn_hash: raise ValueError('Random CNN weights changed')
                verify_originals(snapshot)
                del cache,bundle,cnn; gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
            progress(status='complete',previous_results_verified_unchanged=True)
        except Exception as exc:
            progress(status='failed',error=repr(exc)); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'config.json'))
    parser.add_argument('--run-dir',default=str(PROJECT/'runs/random-cnn'))
    parser.add_argument('--output',default=str(PROJECT/'results/random-cnn'))
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--limit',type=int)
    args=parser.parse_args(); run(args.config,args.run_dir,args.output,torch.device(args.device),args.limit)


if __name__=='__main__': main()
