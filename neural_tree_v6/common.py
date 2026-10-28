"""Shared imports and reproducible configuration for the augmentation follow-up."""
from pathlib import Path
import os
import sys

PROJECT=Path(__file__).resolve().parent
WORKSPACE=PROJECT.parent
os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'runs/matplotlib-cache'))
for directory in ['neural_tree','neural_tree_v2','neural_tree_v3']:
    sys.path.insert(0,str(WORKSPACE/directory))

import numpy as np
import torch
from torch.nn import functional as F
from study import FeatureNode,FeatureForest,SharedCNN,root_scores
from width32_followup import classification_refit
from v2_tree.common import (cpu_state,digest,file_hash,load_checkpoint,read_json,
    restore_rng,rng_state,save_checkpoint,seed_everything,study_lock,tensor_hash,write_json)
from v2_tree.models import correction_targets


def probabilities(scores,classes):
    if classes==2:
        p=scores.sigmoid()
        return torch.stack([1-p,p],1)
    return scores.softmax(1)


def class_loss(scores,y,classes):
    return F.binary_cross_entropy_with_logits(scores,y.float()) if classes==2 else F.cross_entropy(scores,y)


def fixed_json(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Configuration changed: {path}. Use a new run directory.')
    else:write_json(path,value)


def active_nodes(forest):
    nodes=[]
    def visit(node):
        nodes.append(node)
        if node.A is not None and float(node.alpha.detach())>0:visit(node.A)
        if node.B is not None and float(node.beta.detach())>0:visit(node.B)
    for root in forest.roots:visit(root)
    return nodes


def capacity(forest,cnn_parameters):
    walked=[node for root in forest.roots for _,_,node in root.walk()]
    active=active_nodes(forest)
    parameter_count=lambda ns:sum(sum(p.numel() for p in n.base.parameters())+3 for n in ns)
    return {'tree_parameters':parameter_count(walked),'active_tree_parameters':parameter_count(active),
        'nodes':len(walked),'active_nodes':len(active),'correction_nodes':len(walked)-len(forest.roots),
        'maximum_depth':max(depth for root in forest.roots for _,depth,_ in root.walk()),
        'shared_cnn_and_head_parameters':cnn_parameters,
        'total_parameters':cnn_parameters+parameter_count(walked),
        'active_total_parameters':cnn_parameters+parameter_count(active)}
