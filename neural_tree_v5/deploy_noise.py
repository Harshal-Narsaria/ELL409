"""Compact image deployment of a selected label-noise correction forest."""
from pathlib import Path
import argparse
import copy
import torch
from torch import nn
import noise_study as s


class ImageClassifier(nn.Module):
    def __init__(self, cnn, forest, image_mean, image_std, feature_mean, feature_std):
        super().__init__()
        self.cnn, self.forest = cnn, forest
        for p in cnn.parameters(): p.requires_grad_(False)
        for name, value in [('image_mean', image_mean), ('image_std', image_std),
                            ('feature_mean', feature_mean), ('feature_std', feature_std)]:
            self.register_buffer(name, value.clone())

    def forward(self, image):
        normalized = (image-self.image_mean[None,:,None,None])/self.image_std[None,:,None,None]
        x = (self.cnn.features(normalized)-self.feature_mean)/self.feature_std
        return self.forest(x)


def compact_forest(forest):
    result = copy.deepcopy(forest).cpu()
    def prune(node):
        for side, strength in [('A', node.alpha), ('B', node.beta)]:
            child = getattr(node, side)
            if child is None: continue
            if float(strength.detach()) <= 0: setattr(node, side, None)
            else: prune(child)
    for root in result.roots: prune(root)
    return result


def load_deployment(path, device='cpu'):
    saved = s.load_checkpoint(path)
    cnn = s.shared.SharedCNN(saved['classes'])
    cnn.head = nn.Identity()
    cnn.load_state_dict(saved['cnn_model'])
    forest = s.shared.FeatureForest.from_spec(saved['forest_spec'])
    forest.load_state_dict(saved['forest_model'])
    model = ImageClassifier(cnn, forest, saved['image_mean'], saved['image_std'], saved['feature_mean'], saved['feature_std'])
    return model.to(device).eval(), saved


def export_deployment(dataset, source_stage, destination):
    stage = Path(source_stage)
    record = s.read_json(stage/'metrics.json')
    saved = s.load_checkpoint(stage/'forest.pt')
    forest = s.shared.FeatureForest.from_spec(saved['spec'])
    forest.load_state_dict(saved['model']); forest.eval()
    compact = compact_forest(forest).eval()
    cfg = s.read_json(s.PROJECT/'config.json')
    bundle, metadata = s.source_bundle(dataset, cfg)
    x = bundle['splits']['train']['x'][:256]
    with torch.no_grad():
        if not torch.equal(forest(x), compact(x)): raise ValueError('Compacting changed cached-feature scores.')
    folder = s.WORKSPACE/cfg['feature_source']/dataset
    cnn_saved = s.load_checkpoint(folder/'cnn-frozen.pt')
    pixels = s.load_checkpoint(folder/'pixels.pt')
    model_state = {key: value for key, value in cnn_saved['model'].items() if not key.startswith('head.')}
    payload = {'classes': bundle['classes'], 'resolution': 32, 'cnn_model': model_state,
        'forest_spec': compact.spec(), 'forest_model': s.cpu_state(compact),
        'image_mean': pixels['mean'], 'image_std': pixels['std'],
        'feature_mean': bundle['feature_mean'], 'feature_std': bundle['feature_std'],
        'source_metrics': record, 'source_feature_extractor': metadata,
        'allocated_forest_parameters': record['size']['parameters'],
        'compact_forest_parameters': compact.size()['parameters'], 'compact_score_parity': True}
    s.save_checkpoint(destination, payload)
    restored, _ = load_deployment(destination)
    with torch.no_grad():
        if not torch.equal(restored.forest(x), forest(x)): raise ValueError('Deployment reload changed scores.')
        raw = pixels['splits']['train']['pixels'][:16].float()/255
        scores = restored(raw)
        expected = compact(x[:16])
        got = (scores >= 0).long() if bundle['classes']==2 else scores.argmax(1)
        want = (expected >= 0).long() if bundle['classes']==2 else expected.argmax(1)
        disagreement = int((got != want).sum())
        if not torch.isfinite(scores).all() or disagreement:
            raise ValueError('Direct image predictions failed the cached-feature parity check.')
    return {'dataset': dataset, 'source_stage': str(stage), 'total_compact_parameters': sum(p.numel() for p in restored.parameters()),
        'bytes': Path(destination).stat().st_size, 'cached_score_parity': True,
        'direct_image_prediction_disagreements_16_examples': disagreement, 'image_shape': [16,3,32,32]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['cifar10','cats_vs_dogs'])
    parser.add_argument('stage',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(export_deployment(args.dataset,args.stage,args.destination))
