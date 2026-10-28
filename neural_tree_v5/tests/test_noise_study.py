from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
import noise_study as s


def fixture_bundle():
    s.seed_everything(2)
    x = torch.randn(24, 4)
    y = torch.tensor([0, 1] * 12)
    return {'signature': 'fixture', 'classes': 2, 'dimension': 4,
        'splits': {key: {'x': x.clone(), 'y': y.clone(), 'ids': [f'{key}-{i}' for i in range(24)]}
                   for key in ('train', 'validation', 'test')}}


def config():
    return {'seed': 409, 'repeat_seeds': [410, 411], 'widths': [4, 8], 'depth': 1,
        'root_epochs': 3, 'mlp_epochs': 3, 'mlp_batch': 8, 'eval_batch': 16,
        'lr': .001, 'correction_epochs': 2, 'noise_rates': [.15, 0.]}


@pytest.mark.parametrize('classes', [2, 10])
def test_corruption_reproducible_wrong_labels_only_and_no_mutation(classes):
    clean = torch.arange(100) % classes
    before = clean.clone()
    x = torch.arange(100).float()[:, None]
    rng_before = torch.get_rng_state().clone()
    a, metadata = s.corrupt_labels(clean, x, classes, .15, 88)
    b, _ = s.corrupt_labels(clean, x, classes, .15, 88)
    assert torch.equal(a, b) and torch.equal(clean, before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert int((a != clean).sum()) == 15 == metadata['changed_examples']
    assert int(a.min()) >= 0 and int(a.max()) < classes
    zero, _ = s.corrupt_labels(clean, x, classes, 0., 88)
    assert torch.equal(zero, clean)


def test_duplicate_features_share_corruption_to_keep_interpolation_possible():
    x = torch.tensor([[0.], [0.], [1.], [2.], [3.], [3.]])
    y = torch.tensor([0, 0, 1, 0, 1, 1])
    noisy, m = s.corrupt_labels(y, x, 2, .5, 8)
    assert noisy[0] == noisy[1] and noisy[4] == noisy[5]
    assert m['unique_feature_groups'] == 4 and m['changed_groups'] == 2
    with pytest.raises(ValueError, match='conflicting'):
        s.corrupt_labels(torch.tensor([0, 1, 1, 0, 1, 1]), x, 2, .5, 8)


def test_noisy_bundle_leaves_eval_and_source_labels_untouched_and_identity_locked(tmp_path):
    original = fixture_bundle()
    before = {k: v['y'].clone() for k, v in original['splits'].items()}
    noisy = s.noisy_bundle(original, .5, 99, tmp_path)
    assert not torch.equal(noisy['splits']['train']['y'], before['train'])
    for key in ('validation', 'test'):
        assert torch.equal(noisy['splits'][key]['y'], before[key])
    for key in before: assert torch.equal(original['splits'][key]['y'], before[key])
    assert torch.equal(noisy['clean_train_labels'], before['train'])
    assert s.noisy_bundle(original, .5, 99, tmp_path)['signature'] == noisy['signature']
    with pytest.raises(ValueError, match='definition changed'):
        s.noisy_bundle(original, .5, 100, tmp_path)


def test_repeat_selection_independent_of_held_out_metrics():
    rows = [{'width': w, 'train': {'errors': e}, 'test': {'accuracy': a}}
            for w, e, a in [(2, 5, .9), (4, 1, .8), (8, 0, .7), (16, 0, .6)]]
    before = s.representative_widths(rows, [2, 4, 8, 16])
    for r in rows: r['test']['accuracy'] = 1-r['test']['accuracy']
    assert before == s.representative_widths(rows, [2, 4, 8, 16]) == [4, 8, 16]


def test_fixed_topology_inactive_capacity_and_root_ablation(tmp_path):
    cfg = config(); x = torch.randn(10, 4)
    trainer = s.FixedTrainer(tmp_path, cfg, x, 'fixture', 1, 409, 4, lambda **fields: None)
    root = trainer.grow(torch.arange(10), torch.zeros(10))
    forest = s.shared.FeatureForest(2, [root])
    size = s.capacity(forest, {'0': trainer.tasks})
    assert size['nodes'] == 3 and size['parameters'] == 3*((4+2)*4+4)
    assert size['trained_nodes'] <= 3 and size['active_parameters'] <= size['allocated_parameters']
    baseline = s.roots_only(forest)
    assert len(list(baseline.roots[0].walk())) == 1
    assert torch.equal(baseline(x), root.base(x))


def test_resume_matches_uninterrupted_training_and_restores_optimizer_rng(tmp_path):
    cfg = config(); original = fixture_bundle()
    bundle = s.noisy_bundle(original, .25, 99, tmp_path/'labels')
    progress_calls = []
    def interrupt(**fields):
        progress_calls.append(fields)
        if fields.get('epoch') == 1: raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError, match='simulated'):
        s.fit_stage(tmp_path/'resumed', bundle, cfg, 409, 4, interrupt, torch.device('cpu'), 100)
    checkpoint = s.load_checkpoint(tmp_path/'resumed/seed-409/width-4/bases/class-1/root.pt')
    assert checkpoint['epoch'] == 1 and 'optimizer' in checkpoint and 'rng' in checkpoint
    resumed = s.fit_stage(tmp_path/'resumed', bundle, cfg, 409, 4, lambda **fields: None, torch.device('cpu'), 100)
    full = s.fit_stage(tmp_path/'full', bundle, cfg, 409, 4, lambda **fields: None, torch.device('cpu'), 100)
    assert resumed['train']['errors'] == full['train']['errors']
    a = s.load_checkpoint(tmp_path/'resumed/seed-409/width-4/forest.pt')
    b = s.load_checkpoint(tmp_path/'full/seed-409/width-4/forest.pt')
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert 'test' not in resumed and 'test' not in resumed['root_only']
    assert s.fit_stage(tmp_path/'resumed', bundle, cfg, 409, 4, lambda **fields: None, torch.device('cpu'), 100)['identity'] == resumed['identity']
    with pytest.raises(ValueError, match='configuration changed'):
        s.fit_stage(tmp_path/'resumed', bundle, {**cfg, 'lr': .02}, 409, 4, lambda **fields: None, torch.device('cpu'), 100)


def test_test_evaluation_waits_for_every_predeclared_job(tmp_path):
    cfg = config(); bundle = s.noisy_bundle(fixture_bundle(), .25, 99, tmp_path/'labels')
    s.write_json(tmp_path/'stages/repeat-plan.json', {'widths': [4]})
    with pytest.raises(RuntimeError, match='predeclared'):
        s.finalize_test(tmp_path/'stages', bundle, cfg, torch.device('cpu'))
    for seed, width in [(409, 4), (409, 8), (410, 4), (411, 4)]:
        s.fit_stage(tmp_path/'stages', bundle, cfg, seed, width, lambda **fields: None, torch.device('cpu'), 100)
    s.finalize_test(tmp_path/'stages', bundle, cfg, torch.device('cpu'))
    record = s.read_json(tmp_path/'stages/seed-409/width-4/metrics.json')
    assert record['test']['examples'] == 24 and record['root_only']['test']['examples'] == 24


def test_full_sources_use_original_splits_discard_head_and_preserve_labels():
    cfg = s.read_json(s.PROJECT/'config.json')
    if not all((s.WORKSPACE/cfg['feature_source']/d/'features.pt').exists() for d in cfg['datasets']):
        pytest.skip('Needs the saved CNN80 features from neural_tree_v3 (not included in the repository).')
    for dataset in cfg['datasets']:
        source, metadata = s.source_bundle(dataset, cfg)
        assert source['splits']['train']['x'].shape[1] == 128
        assert metadata['CNN_trained_before_noise_on_original_training_labels']
        noisy, m = s.corrupt_labels(source['splits']['train']['y'], source['splits']['train']['x'], source['classes'], .15, cfg['noise_seed'])
        assert .149 < m['actual_fraction'] < .151
        assert len(noisy) == (45000 if dataset == 'cifar10' else 18608)
