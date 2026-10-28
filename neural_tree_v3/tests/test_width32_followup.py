from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import study as s
from width32_followup import classification_refit, fit_stage, saturated
from v2_tree.train import exact_signed_refit


def test_subunit_margin_refit_fixes_classification_without_unit_margin():
    inputs=torch.tensor([[.01,0.,0.],[-.01,0.,0.],[-2.,1.,0.],[2.,0.,1.]])
    labels=torch.tensor([1.,0.,1.,0.])
    assert exact_signed_refit(inputs,labels) is None
    weights,stats=classification_refit(inputs,labels)
    assert weights is not None and bool((weights[:2]>=0).all())
    scores=inputs[:,0]+weights[0]*inputs[:,1]-weights[1]*inputs[:,2]+weights[2]
    assert torch.equal((scores>=0).float(),labels)
    assert 0<stats['maximum_margin_up_to_one']<1


def test_refit_rejects_conflicting_indistinguishable_examples():
    weights,stats=classification_refit(torch.tensor([[0.,0.,0.],[0.,0.,0.]]),torch.tensor([0.,1.]))
    assert weights is None and not stats['classification_feasible']


def test_saturation_requires_all_leaf_base_tasks_fit_and_nonbinding_budget():
    saved={'tasks':{'0':{'root':{'errors_before':5},'root.A':{'errors_before':0}}}}
    assert saturated(saved,10)
    assert not saturated(saved,2)
    saved['tasks']['0']['root.A']['errors_before']=1
    assert not saturated(saved,10)


def test_two_arm_growth_reuses_bases_and_resume_preserves_predictions(tmp_path):
    s.seed_everything(1)
    x=torch.randn(24,129); x[:,128]=torch.linspace(-1,1,24)
    y=torch.tensor([0,1]*12)
    bundle={'classes':2,'signature':'fixture','splits':{split:{'x':x,'y':y} for split in ('train','validation','test')}}
    cfg={'seed':410,'width':32,'cnn_max_epochs':2,'cnn_batch':8,'mlp_epochs':2,'mlp_batch':8,
        'lr':.001,'eval_batch':8,'correction_epochs':2,'max_nodes_per_tree':31,'max_correction_parameters':1000000}
    run,source=tmp_path/'run',tmp_path/'source'
    fit_stage(run,source,bundle,cfg,'original',410,1,torch.device('cpu'),lambda **fields:None)
    before={str(p):s.file_hash(p) for p in (run/'shared-bases').rglob('*.pt')}
    record=fit_stage(run,source,bundle,cfg,'classification_refit',410,1,torch.device('cpu'),lambda **fields:None)
    assert all(s.file_hash(Path(p))==expected for p,expected in before.items())
    path=run/'classification_refit/seed-410/depth-1/forest.pt'
    saved=s.load_checkpoint(path); model=s.FeatureForest.from_spec(saved['spec']); model.load_state_dict(saved['model'])
    assert s.evaluate(model,x,y,8)['errors']==record['train']['errors']
    before_file=s.file_hash(path)
    again=fit_stage(run,source,bundle,cfg,'classification_refit',410,1,torch.device('cpu'),lambda **fields:None)
    assert record==again and s.file_hash(path)==before_file


def test_heldout_labels_cannot_change_fitted_model(tmp_path):
    s.seed_everything(9)
    x=torch.randn(12,129); x[:,128]=torch.linspace(-1,1,12); y=torch.tensor([0,1]*6)
    cfg={'seed':410,'width':32,'cnn_max_epochs':2,'cnn_batch':6,'mlp_epochs':2,'mlp_batch':6,
        'lr':.001,'eval_batch':6,'correction_epochs':2,'max_nodes_per_tree':31,'max_correction_parameters':1000000}
    states=[]
    for i in range(2):
        bundle={'classes':2,'signature':'fixture','splits':{'train':{'x':x,'y':y},
            'validation':{'x':x,'y':y if i==0 else 1-y},'test':{'x':x,'y':y if i==0 else 1-y}}}
        fit_stage(tmp_path/f'run{i}',tmp_path/'source',bundle,cfg,'classification_refit',410,2,torch.device('cpu'),lambda **fields:None)
        states.append(s.load_checkpoint(tmp_path/f'run{i}/classification_refit/seed-410/depth-2/forest.pt')['model'])
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(value,states[1][key]) for key,value in states[0].items())
