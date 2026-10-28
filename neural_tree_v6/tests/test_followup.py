from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from common import *
from data import augmented_pixels,stratified_bootstrap,training_rows
from train import fit_base,Trainer,restore_forest
from analysis import majority_vote,brier_decomposition,landscape
from deploy import ImageEnsemble,export_model,load_deployment
from run import make_plan,finalize


def test_image_augmentation_is_id_keyed_reproducible_and_keeps_canonical():
    pixels=torch.arange(3*8*8).reshape(3,8,8).byte()
    assert torch.equal(augmented_pixels(pixels,'x',0,409,2),pixels)
    first=augmented_pixels(pixels,'x',1,409,2)
    augmented_pixels(pixels,'other',2,409,2)
    assert torch.equal(first,augmented_pixels(pixels,'x',1,409,2))
    assert first.dtype==torch.uint8 and first.shape==pixels.shape
    assert not torch.equal(first,pixels)


def test_bootstrap_draws_are_stratified_and_expand_whole_original_view_groups():
    y=torch.tensor([0]*20+[1]*20)
    draws=stratified_bootstrap(y,.5,409)
    assert len(draws)==20 and torch.equal(torch.bincount(y[draws]),torch.tensor([10,10]))
    assert torch.equal(draws,stratified_bootstrap(y,.5,409))
    values=torch.stack([torch.arange(40)[:,None]+100*v for v in range(3)]).float()
    bundle={'train_views':values,'splits':{'train':{'y':y}}}
    x,labels=training_rows(bundle,draws,'augmented')
    assert len(x)==3*len(draws) and torch.equal(labels,y[draws].repeat(3))
    assert torch.equal(x[:,0].reshape(3,-1)%100,draws.expand(3,-1).float())
    repeated,_=training_rows(bundle,draws,'repeat')
    assert torch.equal(repeated[:,0],draws.repeat(3).float())
    assert len(training_rows(bundle,draws,'none')[0])==len(draws)


def test_majority_votes_on_class_labels_not_probability_magnitudes():
    predictions=torch.tensor([[0,2,1],[0,1,0],[1,2,0]])
    labels,counts=majority_vote(predictions,3)
    assert labels.tolist()==[0,2,0] and counts.sum(1).tolist()==[3,3,3]
    # Odd ensembles can still have multiclass ties; smallest class wins.
    assert majority_vote(torch.tensor([[2],[1],[0]]),3)[0].item()==0
    assert majority_vote(torch.tensor([[1],[0]]),2)[0].item()==0


def test_empirical_brier_decomposition_identity_and_zero_variance():
    torch.manual_seed(1);p=torch.randn(5,20,3).softmax(-1);y=torch.arange(20)%3
    r=brier_decomposition(p,y)
    assert abs(r['mean_member_brier']-r['squared_bias_proxy']-r['bootstrap_prediction_variance'])<1e-10
    exact=F.one_hot(y,3).float()[None].repeat(3,1,1)
    z=brier_decomposition(exact,y)
    assert z['squared_bias_proxy']==0 and z['bootstrap_prediction_variance']==0


def fit_fixture(path,epochs,x,y,previous=None):
    seed_everything(409);model=FeatureNode(3,4).base
    config={'epochs':epochs,'batch':8,'lr':.01,'eval_batch':32,'checkpoint_every':2}
    identity={'path':'root.A','class':1,'seed':409}
    fit_base(model,x,torch.arange(len(y)),y,path,config,identity,lambda **f:None,time.monotonic()+120,previous)
    return model,load_checkpoint(path)


def test_exact_task_epoch_continuation_matches_uninterrupted_training(tmp_path):
    x=torch.randn(24,3,generator=torch.Generator().manual_seed(8));y=(x[:,0]>0).float()
    _,short=fit_fixture(tmp_path/'short.pt',2,x,y)
    a,continued=fit_fixture(tmp_path/'continued.pt',4,x,y,tmp_path/'short.pt')
    b,fresh=fit_fixture(tmp_path/'fresh.pt',4,x,y)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert [(h['errors'],h['loss']) for h in continued['history']]==[(h['errors'],h['loss']) for h in fresh['history']]
    assert torch.equal(continued['rng']['torch'],fresh['rng']['torch'])
    assert load_checkpoint(tmp_path/'short.pt')['epoch']==2
    for key in continued['optimizer']['state']:
        assert all(torch.equal(v,fresh['optimizer']['state'][key][k]) for k,v in continued['optimizer']['state'][key].items())


def test_changed_correction_target_cannot_reuse_optimizer_history(tmp_path):
    x=torch.randn(24,3,generator=torch.Generator().manual_seed(8));y=(x[:,0]>0).float()
    fit_fixture(tmp_path/'original.pt',2,x,y)
    a,_=fit_fixture(tmp_path/'changed.pt',4,x,1-y,tmp_path/'original.pt')
    b,_=fit_fixture(tmp_path/'fresh.pt',4,x,1-y)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    with pytest.raises(ValueError):fit_fixture(tmp_path/'changed.pt',4,x,y)


def test_correction_masks_signs_and_feature_column_isolation():
    p=torch.tensor([0,1,0,1]);y=torch.tensor([0,1,1,0])
    tasks=correction_targets(p,y)
    assert tasks['A'][1].tolist()==[True,False,True,True]
    assert tasks['B'][1].tolist()==[False,True,True,True]
    assert tasks['A'][0].tolist()==[0,0,1,0]
    assert tasks['B'][0].tolist()==[0,0,0,1]
    node=FeatureNode(3,4);x=torch.randn(5,5);changed=x.clone();changed[:,3:]=999
    assert torch.equal(node.base(x),node.base(changed))
    base=node(x).detach();node.A=FeatureNode(3,4)
    with torch.no_grad():
        for parameter in node.A.base.parameters():parameter.zero_()
        node.A.base.layers[-1].bias.fill_(1);node.alpha.fill_(2)
    assert torch.allclose(node(x),base+2)
    node.B=FeatureNode(3,4)
    with torch.no_grad():
        for parameter in node.B.base.parameters():parameter.zero_()
        node.B.base.layers[-1].bias.fill_(1);node.beta.fill_(3)
    assert torch.allclose(node(x),base-1)


def test_standalone_deployment_shares_one_cnn_and_matches_cached_votes(tmp_path):
    seed_everything(3);cnn=SharedCNN(2).eval();forests=[]
    for _ in range(3):
        node=FeatureNode(128,4,128);node.A=FeatureNode(128,4)
        with torch.no_grad():node.alpha.fill_(2)
        forests.append(FeatureForest(2,[node]))
    bundle={'classes':2,'cnn_model':cpu_state(cnn),'image_mean':torch.zeros(3),'image_std':torch.ones(3),
        'feature_mean':torch.zeros(128),'feature_std':torch.ones(128)}
    saved=[{'spec':f.spec(),'model':cpu_state(f)} for f in forests]
    path=tmp_path/'deployment.pt';export_model(bundle,saved,path,{})
    model,_=load_deployment(path);images=torch.rand(3,3,32,32)
    with torch.no_grad():
        z=cnn.features(images);features=torch.cat([z,root_scores(cnn.head(z))],1)
        expected,_=majority_vote(torch.stack([f.predict(features) for f in forests]),2)
        assert torch.equal(model.predict(images),expected)
    assert all(not p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in cnn.parameters())+sum(sum(p.numel() for p in f.parameters()) for f in forests)


def test_loss_slice_center_and_source_weights_are_preserved(tmp_path):
    seed_everything(4);root=FeatureNode(128,4,128);root.A=FeatureNode(128,4)
    with torch.no_grad():root.alpha.fill_(2)
    forest=FeatureForest(2,[root]);x=torch.randn(20,129);y=torch.arange(20)%2
    before=cpu_state(forest)
    landscape(forest,x,y,x,y,tmp_path,{'seed':409,'examples':12,'grid':3,'radius':.1},time.monotonic()+120,lambda **f:None)
    raw=np.load(tmp_path/'landscape.npz');ti=torch.from_numpy(raw['train_indices'])
    with torch.no_grad():expected=float(class_loss(forest(x[ti]),y[ti],2))
    assert np.isclose(raw['losses'][0,1,1],expected) and raw['gate_flip'][1,1]==0
    assert np.isfinite(raw['losses']).all()
    assert all(torch.equal(v,forest.state_dict()[k]) for k,v in before.items())


def test_plan_covers_requirements_and_debug_cannot_unlock_test(tmp_path,monkeypatch):
    from common import PROJECT
    import run
    cfg=read_json(PROJECT/'config.json');jobs=make_plan(cfg)
    assert {'none','repeat','augmented'}=={j['augmentation'] for j in jobs}
    assert set(cfg['sample_fractions']).issubset({j['sample_fraction'] for j in jobs})
    assert set(cfg['epochs']).issubset({j['epochs'] for j in jobs})
    assert set(cfg['depths']).issubset({j['depth'] for j in jobs})
    assert set(cfg['widths']).issubset({j['width'] for j in jobs})
    monkeypatch.setattr(run,'test_features',lambda *a:pytest.fail('Debug test leakage'))
    finalize(tmp_path,tmp_path,{}, {'debug_train_limit':10},[],[],torch.device('cpu'))


def test_reference_test_gate_requires_every_fixed_model(tmp_path,monkeypatch):
    import run
    cfg={'datasets':['fixture'],'seed':409,'depth_width':4,'capacity_depth':1,'capacity_epochs':2,
         'feature_source':'unused','eval_batch':16}
    jobs=[{'dataset':'fixture','augmentation':arm,'noise':0.,'width':4,'depth':1,'epochs':2,
           'sampler':'full','sample_fraction':1.,'replicate':0,'seed':409} for arm in ['none','repeat','augmented']]
    x=torch.zeros(4,129);x[:,128]=torch.tensor([-1.,1.,-2.,2.]);y=torch.tensor([0,1,0,1]);calls=[]
    monkeypatch.setattr(run,'test_features',lambda *a:(calls.append(True) or {'x':x,'y':y}))
    for i,j in enumerate(jobs):
        folder=run.model_folder(tmp_path,j);folder.mkdir(parents=True,exist_ok=True)
        forest=FeatureForest(2,[FeatureNode(128,4,128)])
        save_checkpoint(folder/'model.pt',{'spec':forest.spec(),'model':cpu_state(forest)})
        save_checkpoint(folder/'predictions.pt',{})
        if i<2:write_json(folder/'metrics.json',{})
    assert not run.evaluate_reference(tmp_path,tmp_path/'out',{},cfg,jobs,torch.device('cpu'))
    assert not calls
    write_json(run.model_folder(tmp_path,jobs[-1])/'metrics.json',{})
    assert run.evaluate_reference(tmp_path,tmp_path/'out',{'fixture':{}},cfg,jobs,torch.device('cpu'))
    assert len(calls)==1
    rows=read_json(tmp_path/'out/reference-test.json')['records']
    assert all(r['test']['accuracy']==1 for r in rows)
