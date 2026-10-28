from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import random_cnn_study as study
from v2_tree.common import cpu_state,file_hash,load_checkpoint,save_checkpoint,seed_everything


def small_config():
    return {'feature_seed':409,'seed':409,'channels':[16,32,64],'spatial_grid':2,
        'root_width':8,'root_epochs':2,'mlp_epochs':2,'mlp_batch':6,'lr':.001,'eval_batch':6,
        'correction_epochs':2,'feature_batch':4,'max_nodes_per_tree':31,'max_tree_parameters':1000000}


def test_random_cnn_is_frozen_and_outputs_256_features_without_head():
    seed_everything(1)
    cnn=study.RandomCNN()
    assert cnn(torch.randn(3,3,32,32)).shape==(3,256)
    assert not hasattr(cnn,'head') and all(not p.requires_grad for p in cnn.parameters())
    x=torch.randn(2,3,32,32); before=cpu_state(cnn)
    cnn.train(); a=cnn(x); cnn.eval(); b=cnn(x)
    assert torch.equal(a,b) and all(torch.equal(v,cnn.state_dict()[k]) for k,v in before.items())


def test_random_checkpoint_resume_restores_exact_weights(tmp_path):
    cfg=small_config()
    a,first=study.random_extractor(tmp_path,cfg,torch.device('cpu'))
    b,second=study.random_extractor(tmp_path,cfg,torch.device('cpu'))
    assert first==second
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert load_checkpoint(tmp_path/'random-cnn.pt')['supervised_updates']==0


def test_feature_statistics_use_training_images_only(tmp_path):
    class TinyFeatures(nn.Module):
        dimension=3
        def forward(self,x): return x.mean((2,3))
    cfg=small_config(); train=torch.arange(6*3*4*4).reshape(6,3,4,4).to(torch.uint8)
    stats=[]
    for version in (0,1):
        cache={'fingerprint':'fixture','classes':['cat','dog'],'mean':torch.zeros(3),'std':torch.ones(3),
            'splits':{'train':{'pixels':train,'y':torch.tensor([0,1]*3),'ids':list(range(6))},
            'validation':{'pixels':torch.full((2,3,4,4),version*255,dtype=torch.uint8),'y':torch.tensor([version,version]),'ids':['v1','v2']},
            'test':{'pixels':torch.full((2,3,4,4),version*255,dtype=torch.uint8),'y':torch.tensor([version,version]),'ids':['t1','t2']}}}
        stats.append(study.feature_bundle(TinyFeatures(),cache,tmp_path/str(version),'same-random-cnn',cfg,torch.device('cpu')))
    assert torch.equal(stats[0]['feature_mean'],stats[1]['feature_mean'])
    assert torch.equal(stats[0]['feature_std'],stats[1]['feature_std'])
    assert torch.equal(stats[0]['splits']['train']['x'],stats[1]['splits']['train']['x'])


def test_fixed_roots_are_identical_across_widths_and_depth(tmp_path):
    seed_everything(7); x=torch.randn(18,4); y=torch.tensor([0,1]*9)
    cfg=small_config(); bundle={'dimension':4,'classes':2,'signature':'fixture',
        'splits':{split:{'x':x,'y':y} for split in ('train','validation','test')}}
    study.fit_stage(tmp_path,bundle,cfg,409,4,1,'primary',torch.device('cpu'),lambda **fields:None)
    root=tmp_path/'roots/seed-409/class-1.pt'; before=file_hash(root)
    study.fit_stage(tmp_path,bundle,cfg,409,16,1,'primary',torch.device('cpu'),lambda **fields:None)
    assert file_hash(root)==before
    a=load_checkpoint(tmp_path/'stages/seed-409/width-4/depth-1/forest.pt')
    b=load_checkpoint(tmp_path/'stages/seed-409/width-16/depth-1/forest.pt')
    for key,value in a['model'].items():
        if key.startswith('roots.0.base.'): assert torch.equal(value,b['model'][key])
    assert a['spec']['roots'][0]['width']==b['spec']['roots'][0]['width']==8
    assert set(a['tasks']['1'])==set(b['tasks']['1'])
    for path in a['tasks']['1']: assert a['tasks']['1'][path]['examples']==b['tasks']['1'][path]['examples']
    study.fit_stage(tmp_path,bundle,cfg,409,4,2,'deeper',torch.device('cpu'),lambda **fields:None)
    assert file_hash(root)==before


def test_deployment_predictions_equal_cached_features_and_roundtrip(tmp_path):
    seed_everything(9); cnn=study.RandomCNN()
    root=study.shared.FeatureNode(256,8); root.A=study.shared.FeatureNode(256,4)
    with torch.no_grad(): root.alpha.fill_(2.)
    forest=study.RandomForest(2,[root]); mean,std=torch.zeros(3),torch.ones(3)
    fm,fs=torch.zeros(256),torch.ones(256)
    model=study.ImageClassifier(cnn,forest,mean,std,fm,fs).eval()
    image=torch.rand(3,3,32,32)
    assert torch.equal(model(image),forest(cnn(image)))
    path=tmp_path/'deployment.pt'
    save_checkpoint(path,{'channels':[16,32,64],'grid':2,'cnn_model':cpu_state(cnn),
        'forest_spec':forest.spec(),'forest_model':cpu_state(forest),
        'image_mean':mean,'image_std':std,'feature_mean':fm,'feature_std':fs})
    restored,_=study.load_deployment(path)
    assert torch.equal(model(image),restored(image))
    assert all(not p.requires_grad for p in restored.cnn.parameters())
    assert all(not p.requires_grad for p in restored.forest.roots[0].base.parameters())
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in cnn.parameters())+forest.size()['parameters']


def test_inactive_subtree_is_not_counted_as_active_capacity():
    root=study.shared.FeatureNode(4,8); root.A=study.shared.FeatureNode(4,16)
    forest=study.RandomForest(2,[root]); size=forest.size()
    assert size['nodes']==2 and size['active_nodes']==1
    assert size['active_parameters']<size['parameters']


def test_repeat_selection_cannot_use_test_accuracy():
    cfg={'seed':409}
    rows=[{'seed':409,'experiment':'primary','width':w,'depth_limit':1,'total_parameters':w,
        'train':{'errors':e},'test':{'accuracy':a}} for w,e,a in [(4,5,.9),(8,1,.8),(16,0,.7),(32,0,.6)]]
    before=study.representative_configs(rows,cfg,'primary')
    for row in rows: row['test']['accuracy']=1-row['test']['accuracy']
    assert before==study.representative_configs(rows,cfg,'primary')
    assert before==[{'width':8,'depth':1},{'width':16,'depth':1},{'width':32,'depth':1}]
