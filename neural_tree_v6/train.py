"""Resumable hard error-correction trees with a shared, frozen CNN root."""
import time
from common import *


class BudgetExpired(Exception):pass


def node_metrics(model,x,indices,y,batch):
    model.eval()
    with torch.no_grad():
        scores=torch.cat([model(x[i]) for i in indices.split(batch)])
        return int(((scores>=0)!=y.bool()).sum()),float(F.binary_cross_entropy_with_logits(scores,y))


def fit_base(model,x,indices,y,path,config,identity,progress,deadline,previous=None):
    local={key:config[key] for key in ['epochs','batch','lr','eval_batch','checkpoint_every']}
    task=tensor_hash(indices,y)
    signature=digest({'local':local,'identity':identity,'task':task})
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=0)
    start=0;history=[]
    best_key=node_metrics(model,x,indices,y,config['eval_batch']);best=cpu_state(model)
    saved=None
    if path.exists():
        saved=load_checkpoint(path)
        if saved['signature']!=signature:raise ValueError(f'Resume task/config changed: {path}')
    elif previous and previous.exists():
        candidate=load_checkpoint(previous)
        old_local={**local,'epochs':candidate['local']['epochs']}
        expected=digest({'local':old_local,'identity':identity,'task':task})
        if candidate['signature']==expected and candidate['epoch']<=config['epochs']:
            saved=candidate
    if saved:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        start=saved['epoch'];history=saved['history'];best_key=tuple(saved['best_key']);best=saved['best_model']
    positives=float(y.sum());weight=torch.tensor((len(y)-positives)/positives if 0<positives<len(y) else 1.,device=y.device)
    if time.monotonic()>=deadline:raise BudgetExpired()
    for epoch in range(start,config['epochs']):
        before=time.perf_counter();model.train()
        order=torch.randperm(len(y),device=y.device)
        for positions in order.split(config['batch']):
            optimizer.zero_grad(set_to_none=True)
            scores=model(x[indices[positions]])
            loss=F.binary_cross_entropy_with_logits(scores,y[positions],pos_weight=weight)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite node loss.')
            loss.backward();optimizer.step()
        key=node_metrics(model,x,indices,y,config['eval_batch'])
        if key<best_key:best_key,best=key,cpu_state(model)
        history.append({'epoch':epoch+1,'errors':key[0],'loss':key[1],'seconds':time.perf_counter()-before})
        expired=time.monotonic()>=deadline
        if (epoch+1)%config['checkpoint_every']==0 or epoch+1==config['epochs'] or expired:
            save_checkpoint(path,{'signature':signature,'local':local,'identity':identity,'task':task,
                'epoch':epoch+1,'model':cpu_state(model),'optimizer':optimizer.state_dict(),'rng':rng_state(),
                'best_key':best_key,'best_model':best,'history':history})
            progress(node=identity['path'],class_index=identity['class'],epoch=epoch+1,epochs=config['epochs'],
                     node_examples=len(y),node_errors=key[0],epoch_seconds=history[-1]['seconds'])
        if expired:raise BudgetExpired()
    if not path.exists():
        # A copied exact-task checkpoint can already have the requested budget.
        save_checkpoint(path,{**saved,'signature':signature,'local':local})
    model.load_state_dict(best)
    return {'best_errors':best_key[0],'best_loss':best_key[1],'epoch':config['epochs'],
            'last_errors':history[-1]['errors'],'last_loss':history[-1]['loss'],
            'loss_still_falling':len(history)>=20 and history[-1]['errors']>0 and history[-1]['loss']<.99*history[-20]['loss']}


def fit_parent(inputs,y,path,epochs,identity,deadline):
    signature=digest({'identity':identity,'task':tensor_hash(inputs,y),'epochs':epochs})
    if path.exists():
        done=load_checkpoint(path)
        if done.get('complete'):
            if done['signature']!=signature:raise ValueError('Parent task changed.')
            return done['weights'].to(inputs.device),done['method']
    exact,metadata=classification_refit(inputs,y)
    if exact is not None:
        save_checkpoint(path,{'signature':signature,'complete':True,'weights':exact.cpu(),
                              'method':'verified-positive-margin-LP','metadata':metadata})
        return exact,'verified-positive-margin-LP'
    weights=torch.nn.Parameter(torch.zeros(3,device=inputs.device))
    optimizer=torch.optim.Adam([weights],lr=.05)
    def scores():return inputs[:,0]+weights[0]*inputs[:,1]-weights[1]*inputs[:,2]+weights[2]
    def key():
        with torch.no_grad():
            s=scores();return int(((s>=0)!=y.bool()).sum()),float(F.binary_cross_entropy_with_logits(s,y))
    best_key=key();best=weights.detach().clone();start=0
    if path.exists():
        state=load_checkpoint(path)
        if state['signature']!=signature:raise ValueError('Parent resume identity changed.')
        with torch.no_grad():weights.copy_(state['weights'].to(inputs.device))
        optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        best=state['best'].to(inputs.device);best_key=tuple(state['best_key']);start=state['epoch']
    for epoch in range(start,epochs):
        optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(scores(),y)
        loss.backward();optimizer.step()
        with torch.no_grad():weights[:2].clamp_(min=0)
        current=key()
        if current<best_key:best_key,best=current,weights.detach().clone()
        expired=time.monotonic()>=deadline
        if (epoch+1)%20==0 or epoch+1==epochs or expired:
            save_checkpoint(path,{'signature':signature,'complete':False,'weights':weights.detach().cpu(),
                'optimizer':optimizer.state_dict(),'rng':rng_state(),'best':best.cpu(),'best_key':best_key,'epoch':epoch+1})
        if expired:raise BudgetExpired()
    save_checkpoint(path,{'signature':signature,'complete':True,'weights':best.cpu(),'method':'signed-Adam-fallback'})
    return best,'signed-Adam-fallback'


class Trainer:
    def __init__(self,folder,x,y,classes,config,fingerprint,seed,width,depth,progress,deadline,
                 previous_budget=None,existing=None):
        self.folder,self.x,self.y=folder,x,y
        self.classes,self.config,self.fingerprint=classes,config,fingerprint
        self.seed,self.width,self.depth=seed,width,depth
        self.progress,self.deadline=progress,deadline
        self.previous_budget=previous_budget
        self.existing=existing;self.tasks={}

    def grow(self,indices,y,c,root_index,path='root',level=0):
        if time.monotonic()>=self.deadline:raise BudgetExpired()
        self.unvisited.discard(path);self.used+=1
        seed_everything(self.seed+c+int(digest(path)[:7],16))
        node=FeatureNode(128,self.width,128+root_index if level==0 else None).to(self.x.device)
        identity={'class':c,'path':path,'seed':self.seed,'width':self.width,'features_and_draws':self.fingerprint}
        if level:
            base_path=self.folder/'bases'/f'class-{c}'/(path+'.pt')
            previous=self.previous_budget/'bases'/f'class-{c}'/(path+'.pt') if self.previous_budget else None
            stats=fit_base(node.base,self.x,indices,y,base_path,self.config,identity,self.progress,self.deadline,previous)
        else:stats={'frozen_original_cnn_root':True}
        with torch.no_grad():base=node.base(self.x[indices]);before=(int(((base>=0)!=y.bool()).sum()),float(F.binary_cross_entropy_with_logits(base,y)))
        self.tasks[f'{c}:{path}']={'base':stats,'examples':len(y),'positives':int(y.sum()),'errors_before':before[0]}
        if before[0] and level<self.depth:
            for side,(targets,mask) in correction_targets(base>=0,y).items():
                child_path=path+'.'+side
                room=child_path in self.unvisited or self.used+len(self.unvisited)<self.config['max_nodes_per_tree']
                if room and int(targets[mask].sum())>0:
                    setattr(node,side,self.grow(indices[mask],targets[mask],c,root_index,child_path,level+1))
        if node.A is not None or node.B is not None:
            with torch.no_grad():
                a=(node.A(self.x[indices])>=0).float() if node.A is not None else torch.zeros_like(y)
                b=(node.B(self.x[indices])>=0).float() if node.B is not None else torch.zeros_like(y)
            weights,method=fit_parent(torch.stack([base,a,b],1),y,
                self.folder/f'depth-{self.depth}'/f'class-{c}'/(path+'.pt'),self.config['parent_epochs'],identity,self.deadline)
            with torch.no_grad():
                node.alpha.copy_(weights[0]);node.beta.copy_(weights[1]);node.offset.copy_(weights[2])
                corrected=node(self.x[indices]);after=(int(((corrected>=0)!=y.bool()).sum()),float(F.binary_cross_entropy_with_logits(corrected,y)))
                if after>before:node.alpha.zero_();node.beta.zero_();node.offset.zero_()
            self.tasks[f'{c}:{path}']['parent_method']=method
        with torch.no_grad():self.tasks[f'{c}:{path}']['errors_after']=int(((node(self.x[indices])>=0)!=y.bool()).sum())
        return node

    def fit(self):
        roots=[]
        for i,c in enumerate([1] if self.classes==2 else range(self.classes)):
            self.used=0
            self.unvisited={p for p,_,_ in self.existing.roots[i].walk()} if self.existing else set()
            roots.append(self.grow(torch.arange(len(self.y),device=self.x.device),(self.y==c).float(),c,i).cpu())
        return FeatureForest(self.classes,roots).to(self.x.device)


@torch.no_grad()
def predict(forest,x,batch=2048):
    forest.eval()
    device=next(forest.parameters()).device
    return torch.cat([forest(part.to(device)).cpu() for part in x.split(batch)])


def evaluate(forest,x,y,batch=2048):
    scores=predict(forest,x,batch)
    pred=(scores>=0).long() if forest.classes==2 else scores.argmax(1)
    p=probabilities(scores,forest.classes)
    return {'examples':len(y),'errors':int((pred!=y).sum()),'accuracy':float((pred==y).float().mean()),
        'loss':float(class_loss(scores,y,forest.classes)),
        'brier':float((p-F.one_hot(y,forest.classes)).square().sum(1).mean())},p,pred


def restore_forest(path,device='cpu'):
    saved=load_checkpoint(path)
    forest=FeatureForest.from_spec(saved['spec']);forest.load_state_dict(saved['model'])
    return forest.to(device).eval(),saved
