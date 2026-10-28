from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def correction_targets(prediction, target):
    """Table II in the reference: X entries are excluded by the masks."""
    p, y = prediction.bool(), target.bool()
    return {"A": ((~p & y).float(), ~(p & y)),
            "B": ((p & ~y).float(), ~(~p & ~y))}


class SharedCNN(nn.Module):
    feature_dim = 128

    def __init__(self, classes):
        super().__init__()
        blocks = []
        inputs = 3
        for channels in (32, 64, 128):
            blocks.append(nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                        nn.MaxPool2d(2)))
            inputs = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(128, classes)

    def spatial_features(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def features(self, x):
        return self.spatial_features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.features(x))


class BaseNode(nn.Module):
    def __init__(self, features=128, width=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class CorrectionNode(nn.Module):
    def __init__(self, features=128, width=64, active=True):
        super().__init__()
        self.features, self.width, self.active = features, width, active
        self.base = BaseNode(features, width)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.A = None
        self.B = None

    def forward(self, x):
        if not self.active:
            return torch.full((len(x),), -1.0, device=x.device)
        score = self.base(x) + self.offset
        if self.A is not None and self.A.active:
            score = score + self.alpha.clamp_min(0) * (self.A(x) >= 0).to(x.dtype)
        if self.B is not None and self.B.active:
            score = score - self.beta.clamp_min(0) * (self.B(x) >= 0).to(x.dtype)
        return score

    def spec(self):
        return {"features": self.features, "width": self.width, "active": self.active,
                "A": self.A.spec() if self.A is not None else None,
                "B": self.B.spec() if self.B is not None else None}

    @classmethod
    def from_spec(cls, spec):
        node = cls(spec["features"], spec["width"], spec["active"])
        for side in ("A", "B"):
            if spec[side] is not None:
                setattr(node, side, cls.from_spec(spec[side]))
        return node

    def walk(self, path="root", depth=0):
        yield path, depth, self
        for side in ("A", "B"):
            child = getattr(self, side)
            if child is not None:
                yield from child.walk(path + "." + side, depth + 1)

    def pad(self, depth):
        if depth <= 0:
            return
        for side in ("A", "B"):
            if getattr(self, side) is None:
                setattr(self, side, CorrectionNode(self.features, self.width, active=False))
            getattr(self, side).pad(depth - 1)


class TreeClassifier(nn.Module):
    def __init__(self, classes, roots):
        super().__init__()
        self.classes = classes
        self.roots = nn.ModuleList(roots)

    def forward(self, x):
        scores = torch.stack([root(x) for root in self.roots], dim=1)
        return scores[:, 0] if self.classes == 2 else scores

    def predict(self, x):
        scores = self(x)
        return (scores >= 0).long() if self.classes == 2 else scores.argmax(dim=1)

    def loss(self, x, y):
        scores = self(x)
        # Multiclass CE measures final argmax predictions; training remains one-vs-rest BCE.
        return F.binary_cross_entropy_with_logits(scores, y.float()) if self.classes == 2 else F.cross_entropy(scores, y)

    def spec(self):
        return {"classes": self.classes, "roots": [r.spec() for r in self.roots]}

    @classmethod
    def from_spec(cls, spec):
        return cls(spec["classes"], [CorrectionNode.from_spec(s) for s in spec["roots"]])

    def size(self, trunk_parameters=0):
        allocated = sum(p.numel() for p in self.parameters())
        active_nodes = [n for r in self.roots for _, _, n in r.walk() if n.active]
        active = sum(sum(p.numel() for p in n.base.parameters()) + 3 for n in active_nodes)
        depths = [d for r in self.roots for _, d, n in r.walk() if n.active]
        return {"allocated_parameters": allocated + trunk_parameters,
                "active_parameters": active + trunk_parameters,
                "trunk_parameters": trunk_parameters, "active_nodes": len(active_nodes),
                "allocated_nodes": sum(1 for r in self.roots for _ in r.walk()),
                "max_depth": max(depths, default=0)}


class ImageTreeClassifier(nn.Module):
    """Complete deployment: input RGB tensors in [0,1], preprocessing, CNN, and trees."""
    def __init__(self, tree, bundle):
        super().__init__()
        self.trunk = SharedCNN(tree.classes).blocks
        self.trunk.load_state_dict(bundle["trunk_state"])
        self.tree = tree
        self.register_buffer("image_mean", torch.tensor(bundle["image_mean"])[:, None, None])
        self.register_buffer("image_std", torch.tensor(bundle["image_std"])[:, None, None])
        self.register_buffer("feature_mean", bundle["feature_mean"])
        self.register_buffer("feature_std", bundle["feature_std"])

    def forward(self, image):
        x = (image - self.image_mean) / self.image_std
        for block in self.trunk:
            x = block(x)
        features = (x.mean((2, 3)) - self.feature_mean) / self.feature_std
        return self.tree(features)

    def predict(self, image):
        score = self(image)
        return (score >= 0).long() if self.tree.classes == 2 else score.argmax(1)
