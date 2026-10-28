from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
import noise_study as s
import noise_followup as f
from test_noise_study import fixture_bundle, config


def test_extension_matches_fresh_common_budget_and_preserves_original(tmp_path):
    old, new = config(), {**config(), 'root_epochs': 6, 'mlp_epochs': 6}
    bundle = s.noisy_bundle(fixture_bundle(), .25, 99, tmp_path/'labels')
    device = torch.device('cpu'); callback = lambda **fields: None
    s.fit_stage(tmp_path/'old', bundle, old, 409, 4, callback, device, 100)
    before = {str(p): s.file_hash(p) for p in (tmp_path/'old').rglob('*.pt')}
    with f.continue_matching_nodes(tmp_path/'old', tmp_path/'continued', old):
        s.fit_stage(tmp_path/'continued', bundle, new, 409, 4, callback, device, 100)
    s.fit_stage(tmp_path/'fresh', bundle, new, 409, 4, callback, device, 100)
    a = s.load_checkpoint(tmp_path/'continued/seed-409/width-4/forest.pt')
    b = s.load_checkpoint(tmp_path/'fresh/seed-409/width-4/forest.pt')
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert all(s.file_hash(p) == hash_ for p, hash_ in before.items())
    info = s.read_json(tmp_path/'continued/seed-409/width-4/bases/class-1/root.continuation.json')
    assert info['mode'] == 'continued-exact-task' and info['starting_epoch'] == 3


def test_changed_child_task_cannot_reuse_optimizer(tmp_path):
    old, new = config(), {**config(), 'root_epochs': 6, 'mlp_epochs': 6}
    bundle = s.noisy_bundle(fixture_bundle(), .25, 99, tmp_path/'labels')
    s.fit_stage(tmp_path/'old', bundle, old, 409, 4, lambda **fields: None, torch.device('cpu'), 100)
    path = tmp_path/'old/seed-409/width-4/bases/class-1/root.pt'
    local = {k: new[k] for k in ('root_epochs','mlp_epochs','mlp_batch','lr','eval_batch')}
    local.update(width=4, seed=409)
    identity = {'path':'root','depth':0,'kind':'mlp','class':1,'features_and_labels':bundle['signature']}
    y = (bundle['splits']['train']['y']==1).float()
    outcome = f.fork_checkpoint(path, tmp_path/'new.pt', torch.arange(len(y)), 1-y, local, identity, old)
    assert outcome['mode'] == 'fresh' and not (tmp_path/'new.pt').exists()


def test_interrupted_extension_resumes_without_recopying_source(tmp_path):
    old, new = config(), {**config(), 'root_epochs': 6, 'mlp_epochs': 6}
    bundle = s.noisy_bundle(fixture_bundle(), .25, 99, tmp_path/'labels')
    s.fit_stage(tmp_path/'old', bundle, old, 409, 4, lambda **fields: None, torch.device('cpu'), 100)
    def stop(**fields):
        if fields.get('epoch')==4: raise RuntimeError('interrupt extension')
    with f.continue_matching_nodes(tmp_path/'old', tmp_path/'new', old):
        with pytest.raises(RuntimeError, match='interrupt extension'):
            s.fit_stage(tmp_path/'new', bundle, new, 409, 4, stop, torch.device('cpu'), 100)
    assert s.load_checkpoint(tmp_path/'new/seed-409/width-4/bases/class-1/root.pt')['epoch']==4
    with f.continue_matching_nodes(tmp_path/'old', tmp_path/'new', old):
        result=s.fit_stage(tmp_path/'new', bundle, new, 409, 4, lambda **fields: None, torch.device('cpu'), 100)
    assert result['train']['examples']==24 and 'test' not in result
