from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from study import (FeatureForest, FeatureNode, ImageClassifier, ProgressiveTrainer,
    root_scores, train_to_target)
from v2_tree.common import cpu_state, load_checkpoint, save_checkpoint, seed_everything


def test_root_decisions_match_cnn_binary_multiclass_and_ties():
    logits = torch.tensor([[3.,1.,2.],[1.,4.,2.],[2.,2.,1.],[0.,0.,0.]])
    assert torch.equal(root_scores(logits).argmax(1),logits.argmax(1))
    binary = logits[:,:2]
    assert torch.equal((root_scores(binary).squeeze(1)>=0).long(),binary.argmax(1))


def test_cnn_stops_at_first_crossing_and_resume_cannot_train_further(tmp_path):
    seed_everything(1)
    cnn = nn.Linear(2,2)
    images = torch.tensor([[-1.,0.],[1.,0.]])
    labels = torch.tensor([0,1])
    cfg = {'seed':1,'cnn_target_accuracy':0.,'cnn_max_epochs':5,'cnn_lr':.001,'cnn_batch':2}
    calls=[]
    saved = train_to_target(cnn,images,labels,tmp_path,cfg,'fixture',lambda **f:calls.append(f))
    assert saved['epoch']==1 and len(calls)==1
    before=cpu_state(cnn)
    with torch.no_grad(): cnn.weight.fill_(999)
    reloaded=train_to_target(cnn,images,labels,tmp_path,cfg,'fixture',lambda **f:None)
    assert reloaded['epoch']==1 and all(torch.equal(v,cnn.state_dict()[k]) for k,v in before.items())


def test_progressive_growth_reuses_existing_base_checkpoints(tmp_path):
    seed_everything(2)
    x=torch.randn(20,129); y=torch.tensor([0.,1.]*10)
    x[:10,128]=-1; x[10:,128]=1
    cfg={'seed':409,'width':4,'mlp_epochs':2,'mlp_batch':10,'cnn_epochs':2,'cnn_batch':10,
         'lr':.001,'eval_batch':10,'correction_epochs':2}
    trainer=ProgressiveTrainer(tmp_path,cfg,x,'fixture',1,0,lambda **f:None)
    shallow=trainer.grow(torch.arange(20),y,1,15)
    path=tmp_path/'bases/root.A.pt'
    before=load_checkpoint(path)
    existing=[p for p,_,_ in shallow.walk()]
    trainer=ProgressiveTrainer(tmp_path,cfg,x,'fixture',1,0,lambda **f:None,existing)
    deeper=trainer.grow(torch.arange(20),y,2,5)
    after=load_checkpoint(path)
    assert after['epoch']==before['epoch']==2
    assert all(torch.equal(v,after['model'][k]) for k,v in before['model'].items())
    assert set(existing).issubset({p for p,_,_ in deeper.walk()})
    assert len(list(deeper.walk()))>=len(list(shallow.walk()))
    assert not list(deeper.base.parameters())


def test_correction_features_cannot_read_frozen_root_score_columns():
    from study import FeatureScore
    score = FeatureScore(3,4)
    x=torch.randn(5,6); modified=x.clone(); modified[:,3:]=9999
    assert torch.equal(score(x),score(modified))


def test_direct_image_predictions_match_cached_feature_predictions_and_cnn_frozen():
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__(); self.head=nn.Linear(3,3)
        def features(self,image): return image.mean((2,3))
    cnn=TinyCNN()
    forest=FeatureForest(3,[FeatureNode(3,4,3+i) for i in range(3)])
    forest.roots[1].A=FeatureNode(3,4)
    with torch.no_grad(): forest.roots[1].alpha.fill_(2)
    model=ImageClassifier(cnn,forest,torch.zeros(3),torch.ones(3),torch.zeros(3),torch.ones(3))
    images=torch.rand(5,3,8,8)
    z=cnn.features(images); x=torch.cat([z,root_scores(cnn.head(z))],1)
    assert torch.equal(model(images),forest(x))
    assert all(not p.requires_grad for p in cnn.parameters())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in cnn.parameters())+forest.size()['parameters']


def test_forest_roundtrip_preserves_frozen_roots_and_corrections(tmp_path):
    root=FeatureNode(3,4,3)
    root.A=FeatureNode(3,4)
    with torch.no_grad(): root.alpha.fill_(2)
    model=FeatureForest(2,[root]); x=torch.randn(5,4)
    save_checkpoint(tmp_path/'forest.pt',{'spec':model.spec(),'model':cpu_state(model)})
    saved=load_checkpoint(tmp_path/'forest.pt'); restored=FeatureForest.from_spec(saved['spec'])
    restored.load_state_dict(saved['model'])
    assert torch.equal(model(x),restored(x))
    assert restored.size()['nodes']==2


def test_complete_deployment_roundtrip_includes_one_shared_cnn(tmp_path):
    from study import SharedCNN, load_deployment
    seed_everything(409)
    cnn=SharedCNN(2)
    root=FeatureNode(128,4,128); root.A=FeatureNode(128,4)
    with torch.no_grad(): root.alpha.fill_(2)
    forest=FeatureForest(2,[root])
    mean,std=torch.zeros(3),torch.ones(3)
    fm,fs=torch.zeros(128),torch.ones(128)
    model=ImageClassifier(cnn,forest,mean,std,fm,fs).eval()
    images=torch.rand(3,3,32,32)
    save_checkpoint(tmp_path/'deployment.pt',{'cnn_model':cpu_state(cnn),'forest_spec':forest.spec(),
        'forest_model':cpu_state(forest),'image_mean':mean,'image_std':std,'feature_mean':fm,'feature_std':fs})
    restored,_=load_deployment(tmp_path/'deployment.pt')
    assert torch.equal(model(images),restored(images))
    assert all(not p.requires_grad for p in restored.cnn.parameters())
