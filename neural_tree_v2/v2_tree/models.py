from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def correction_targets(prediction, target):
    """Reference Table II: A addresses false negatives, B false positives."""
    p, y = prediction.bool(), target.bool()
    return {"A": ((~p & y).float(), ~(p & y)), "B": ((p & ~y).float(), ~(~p & ~y))}


class PixelScore(nn.Module):
    def __init__(self, resolution, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(3 * resolution**2, width),
                                    nn.ReLU(), nn.Linear(width, 1))

    def forward(self, image):
        return self.layers(image).squeeze(1)


class ConvScore(nn.Module):
    """A fresh CNN at one node, trained on that node's masked correction task."""
    def __init__(self, width, channels=(8, 16)):
        super().__init__()
        a, b = channels
        self.trunk = nn.Sequential(nn.Conv2d(3, a, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.AdaptiveAvgPool2d((4, 4)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(b * 16, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, image):
        return self.head(self.trunk(image)).squeeze(1)


class ImageNode(nn.Module):
    def __init__(self, resolution=32, width=16, kind="mlp", channels=(8, 16)):
        super().__init__()
        self.resolution, self.width, self.kind, self.channels = resolution, width, kind, tuple(channels)
        self.base = PixelScore(resolution, width) if kind == "mlp" else ConvScore(width, channels)
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.beta = nn.Parameter(torch.tensor(0.))
        self.offset = nn.Parameter(torch.tensor(0.))
        self.A = self.B = None

    def forward(self, image):
        score = self.base(image) + self.offset
        if self.A is not None:
            score = score + self.alpha.clamp_min(0) * (self.A(image) >= 0).to(score.dtype)
        if self.B is not None:
            score = score - self.beta.clamp_min(0) * (self.B(image) >= 0).to(score.dtype)
        return score

    def walk(self, path="root", depth=0):
        yield path, depth, self
        for side in ("A", "B"):
            child = getattr(self, side)
            if child is not None:
                yield from child.walk(path + "." + side, depth + 1)

    def spec(self):
        return {"resolution": self.resolution, "width": self.width, "kind": self.kind,
                "channels": list(self.channels), "A": self.A.spec() if self.A is not None else None,
                "B": self.B.spec() if self.B is not None else None}

    @classmethod
    def from_spec(cls, spec):
        node = cls(spec["resolution"], spec["width"], spec["kind"], spec["channels"])
        for side in ("A", "B"):
            if spec[side] is not None:
                setattr(node, side, cls.from_spec(spec[side]))
        return node


class ImageForest(nn.Module):
    def __init__(self, classes, roots):
        super().__init__()
        self.classes = classes
        self.roots = nn.ModuleList(roots)

    def forward(self, images):
        scores = torch.stack([root(images) for root in self.roots], dim=1)
        return scores[:, 0] if self.classes == 2 else scores

    def predict(self, images):
        scores = self(images)
        return (scores >= 0).long() if self.classes == 2 else scores.argmax(1)

    def loss(self, images, labels):
        scores = self(images)
        return F.binary_cross_entropy_with_logits(scores, labels.float()) if self.classes == 2 else F.cross_entropy(scores, labels)

    def spec(self):
        return {"classes": self.classes, "roots": [root.spec() for root in self.roots]}

    @classmethod
    def from_spec(cls, spec):
        return cls(spec["classes"], [ImageNode.from_spec(s) for s in spec["roots"]])

    def size(self):
        nodes = [(depth, node) for root in self.roots for _, depth, node in root.walk()]
        return {"parameters": sum(p.numel() for p in self.parameters()), "nodes": len(nodes),
                "cnn_nodes": sum(n.kind == "cnn" for _, n in nodes),
                "mlp_nodes": sum(n.kind == "mlp" for _, n in nodes),
                "max_depth": max(d for d, _ in nodes), "frozen_learned_parameters": 0}
