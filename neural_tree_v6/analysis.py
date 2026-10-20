"""Majority voting, empirical Brier decomposition and hard-tree loss slices."""
import copy
from common import *
from train import predict,evaluate


def majority_vote(predictions,classes):
    if predictions.ndim!=2:raise ValueError('Expected members x examples.')
    counts=F.one_hot(predictions.long(),classes).sum(0)
    # Deterministic tie rule: smallest class index. Primary ensemble sizes are odd.
    return counts.argmax(1),counts


def brier_decomposition(member_probabilities,y):
    if len(member_probabilities)<2:raise ValueError('Need multiple independently fitted members.')
    p=member_probabilities.double()
    targets=F.one_hot(y,p.shape[-1]).double()
    average=p.mean(0)
    bias_proxy=(average-targets).square().sum(-1).mean()
    variance=(p-average).square().sum(-1).mean()
    expected_brier=(p-targets).square().sum(-1).mean()
    assert torch.allclose(bias_proxy+variance,expected_brier,atol=1e-10,rtol=1e-8)
    return {'members':len(p),'examples':len(y),'squared_bias_proxy':float(bias_proxy),
        'bootstrap_prediction_variance':float(variance),'mean_member_brier':float(expected_brier),
        'decomposition_residual':float(expected_brier-bias_proxy-variance),
        'variance_denominator':'M (empirical population variance)',
        'scope':'Conditional on fixed pretrained features, fixed evaluation images and observed one-hot labels. This is not bias against an unknown Bayes probability and does not isolate irreducible noise.'}


def disagreement(predictions):
    values=[float((predictions[i]!=predictions[j]).float().mean())
            for i in range(len(predictions)) for j in range(i+1,len(predictions))]
    return sum(values)/len(values) if values else 0.


def normalized_directions(model,seed):
    generator=torch.Generator().manual_seed(seed)
    directions=[]
    # Perturb output-row normalized MLP weight matrices; hold biases and all
    # signed parent strengths fixed. Frozen CNN/root functions are excluded.
    for name,p in model.named_parameters():
        if '.base.' not in name or p.ndim!=2:continue
        pair=[]
        for _ in range(2):
            raw=torch.randn(p.shape,generator=generator,dtype=p.dtype).to(p.device)
            norm=raw.norm(dim=1,keepdim=True).clamp_min(1e-12)
            pair.append(raw/norm*p.detach().norm(dim=1,keepdim=True))
        directions.append((name,p.detach().clone(),pair[0],pair[1]))
    return directions


@torch.no_grad()
def landscape(forest,train_x,train_y,val_x,val_y,folder,config,deadline,progress):
    import time
    model=copy.deepcopy(forest).eval()
    if not normalized_directions(model,config['seed']):return
    generator=torch.Generator().manual_seed(config['seed'])
    ti=torch.randperm(len(train_y),generator=generator)[:config['examples']]
    vi=torch.randperm(len(val_y),generator=generator)[:config['examples']]
    device=next(model.parameters()).device
    xx=[train_x[ti].to(device),val_x[vi].to(device)]
    yy=[train_y[ti].to(device),val_y[vi].to(device)]
    directions=normalized_directions(model,config['seed'])
    params=dict(model.named_parameters());nodes=active_nodes(model)
    gate_reference=torch.stack([n(xx[1])>=0 for n in nodes])
    coordinates=np.linspace(-config['radius'],config['radius'],config['grid'])
    losses=np.full((2,len(coordinates),len(coordinates)),np.nan)
    flips=np.full(losses.shape[1:],np.nan)
    before=cpu_state(model)
    try:
        for i,a in enumerate(coordinates):
            for j,b in enumerate(coordinates):
                if time.monotonic()>=deadline:raise TimeoutError('Landscape reached training-window limit.')
                for name,base,u,v in directions:params[name].copy_(base+a*u+b*v)
                for split in [0,1]:losses[split,i,j]=float(class_loss(model(xx[split]),yy[split],model.classes))
                flips[i,j]=float((torch.stack([n(xx[1])>=0 for n in nodes])!=gate_reference).float().mean())
            progress(stage='loss_landscape',landscape_row=i+1,landscape_rows=len(coordinates))
    finally:model.load_state_dict(before)
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'landscape.npz',coordinates=coordinates,losses=losses,gate_flip=flips,
                        train_indices=ti.numpy(),validation_indices=vi.numpy())
    write_json(folder/'landscape.json',{'grid':config['grid'],'seed':config['seed'],'radius':config['radius'],
        'training_examples':len(ti),'validation_examples':len(vi),'original_model_restored':True,
        'directions':'Output-row norm-matched random directions on base-MLP weight matrices only; biases, roots, CNN and parent correction coefficients fixed.',
        'hard_gates_recomputed':True,'gate_flip':'Fraction of active node decisions differing from reference on the fixed validation slice.',
        'plotted_objective':'Final classifier cross-entropy/BCE, not the sum of masked node-training objectives. The saved point need not be stationary for this global loss.',
        'limitations':'Two random slices, not the global landscape. Hard recursive thresholds create discontinuities; no smooth Hessian or global flatness claim.'})


def summarize(run,output):
    records=[read_json(p) for p in run.glob('models/*/epochs-*/depth-*/metrics.json')]
    records.sort(key=lambda r:(r['dataset'],r['job']['augmentation'],r['job']['sample_fraction'],r['job']['epochs'],r['job']['seed'],r['job']['width'],r['job']['depth']))
    write_json(output/'measurements.json',records)
    grouped={}
    for r in records:
        job=r['job']
        if job['sampler']!='bootstrap' or job['noise']!=0:continue
        key=(r['dataset'],job['augmentation'],job['sample_fraction'],job['epochs'],job['depth'],job['width'])
        grouped.setdefault(key,[]).append(r)
    decompositions=[];ensembles=[]
    for key,rows in grouped.items():
        rows.sort(key=lambda r:r['job']['replicate'])
        arrays=[load_checkpoint(run/r['predictions_path']) for r in rows]
        if len(arrays)<2:continue
        ys=arrays[0]['validation_y']
        assert all(torch.equal(a['validation_y'],ys) for a in arrays)
        probs=torch.stack([a['validation_probabilities'] for a in arrays])
        preds=torch.stack([a['validation_prediction'] for a in arrays])
        detail={'dataset':key[0],'augmentation':key[1],'sample_fraction':key[2],
                'epochs_per_child':key[3],'depth':key[4],'width':key[5],
                'sample_draws':[r['sample_draws'] for r in rows],'unique_originals':[r['unique_originals'] for r in rows],
                'replicates':[r['job']['replicate'] for r in rows]}
        decompositions.append({**detail,**brier_decomposition(probs,ys),
            'hard_prediction_decomposition':brier_decomposition(F.one_hot(preds,rows[0]['classes']).float(),ys)})
        if key[2]==1:
            for count in [1,3,5]:
                if len(rows)<count:continue
                prediction,votes=majority_vote(preds[:count],rows[0]['classes'])
                ensembles.append({**detail,'members':count,'validation_accuracy':float((prediction==ys).float().mean()),
                    'mean_member_accuracy':float((preds[:count]==ys).float().mean()),
                    'pairwise_disagreement':disagreement(preds[:count]),
                    'validation_ties':int((votes==votes.max(1,keepdim=True).values).sum(1).gt(1).sum()),
                    'ensemble_total_parameters':rows[0]['size']['shared_cnn_and_head_parameters']+sum(r['size']['tree_parameters'] for r in rows[:count]),
                    'voting':'One complete classifier contributes one final class label; ties use smallest class index.'})
    write_json(output/'bias-variance.json',decompositions)
    write_json(output/'ensembles.json',ensembles)
    return records,decompositions,ensembles
