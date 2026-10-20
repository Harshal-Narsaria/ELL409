"""Deterministic image augmentation, performed before frozen-CNN extraction."""
import hashlib
from common import *


def augmented_pixels(pixels,sample_id,view,seed,padding=4):
    if view==0:return pixels.clone()
    key=f'{seed}:{sample_id}:{view}'.encode()
    local_seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
    generator=np.random.default_rng(local_seed)
    image=pixels.cpu().numpy()
    _,h,w=image.shape
    padded=np.pad(image,((0,0),(padding,padding),(padding,padding)),mode='reflect')
    top,left=generator.integers(0,2*padding+1,size=2)
    result=padded[:,top:top+h,left:left+w]
    if generator.random()<.5:result=result[:,:,::-1]
    return torch.from_numpy(result.copy())


def stratified_bootstrap(y,fraction,seed):
    if not 0<fraction<=1:raise ValueError('Sample fraction must be in (0,1].')
    generator=torch.Generator().manual_seed(seed)
    draws=[]
    for label in y.unique(sorted=True):
        members=torch.where(y==label)[0]
        n=max(1,round(len(members)*fraction))
        draws.append(members[torch.randint(len(members),(n,),generator=generator)])
    selected=torch.cat(draws)
    return selected[torch.randperm(len(selected),generator=generator)]


def balanced_prefix(y,limit):
    if limit is None:return torch.arange(len(y))
    if limit<len(y.unique()):raise ValueError('Debug subset must contain all classes.')
    groups=[torch.where(y==label)[0] for label in y.unique(sorted=True)]
    result=[]
    for i in range(max(map(len,groups))):
        for group in groups:
            if i<len(group):result.append(group[i])
            if len(result)==min(limit,len(y)):return torch.stack(result)
    return torch.stack(result)


def prepare(dataset,folder,config,device,progress):
    seed_everything(config['seed'])
    source=WORKSPACE/config['feature_source']/dataset
    source_paths={name:source/name for name in ['features.pt','pixels.pt','cnn-frozen.pt']}
    manifest_path=WORKSPACE/'neural_tree/data'/dataset/'manifest.json'
    identity={'dataset':dataset,'source':{k:file_hash(p) for k,p in source_paths.items()},
        'manifest_sha256':file_hash(manifest_path),'views':config['views'],'augmentation_version':'reflect-crop-flip-v1',
        'augmentation_seed':config['augmentation_seed'],'padding':config['padding'],
        'debug_train_limit':config.get('debug_train_limit'),'debug_validation_limit':config.get('debug_validation_limit')}
    fixed_json(folder/'feature-definition.json',identity)
    signature=digest(identity)
    base_path=folder/'canonical.pt'
    if base_path.exists():
        base=load_checkpoint(base_path)
        if base['signature']!=signature:raise ValueError('Feature identity changed.')
        if all((folder/f'train-view-{v}.pt').exists() for v in range(config['views'])):return load_bundle(folder,config)
    saved=load_checkpoint(source_paths['features.pt'])
    pixels=load_checkpoint(source_paths['pixels.pt'])
    checkpoint=load_checkpoint(source_paths['cnn-frozen.pt'])
    manifest=read_json(manifest_path)
    cnn=SharedCNN(saved['classes']).to(device).eval();cnn.load_state_dict(checkpoint['model'])
    for p in cnn.parameters():p.requires_grad_(False)
    select={split:balanced_prefix(saved['splits'][split]['y'],config.get('debug_'+split+'_limit'))
            for split in ['train','validation']}
    for split in select:
        expected=manifest['splits'][split]
        assert saved['splits'][split]['ids']==[r['id'] for r in expected]
        assert pixels['splits'][split]['ids']==saved['splits'][split]['ids']
        assert torch.equal(saved['splits'][split]['y'],torch.tensor([r['label'] for r in expected]))
    base={'signature':signature,'classes':saved['classes'],'dimension':128,
        'feature_mean':saved['feature_mean'],'feature_std':saved['feature_std'],
        'image_mean':pixels['mean'],'image_std':pixels['std'],
        'cnn_model':checkpoint['model'],'cnn_parameters':sum(p.numel() for p in cnn.parameters()),
        'cnn_training_epoch':checkpoint['epoch'],'splits':{},'debug_subset':config.get('debug_train_limit') is not None,
        'source_feature_sha256':identity['source']['features.pt'],
        'frozen_features_were_trained_on_original_full_training_pool':True,'test_evaluated':False}
    for split,indices in select.items():
        data=saved['splits'][split]
        base['splits'][split]={'x':data['x'][indices].clone(),'y':data['y'][indices].clone(),
                             'ids':[data['ids'][i] for i in indices.tolist()]}
    save_checkpoint(base_path,base)
    train_pixels=pixels['splits']['train']['pixels'][select['train']]
    ids=base['splits']['train']['ids']
    for view in range(config['views']):
        destination=folder/f'train-view-{view}.pt'
        if destination.exists():
            assert load_checkpoint(destination)['signature']==signature
            continue
        progress(stage='extracting_augmented_features',dataset=dataset,view=view)
        if view==0:features=base['splits']['train']['x']
        else:
            output=[]
            with torch.no_grad():
                for start in range(0,len(ids),config['feature_batch']):
                    transformed=torch.stack([augmented_pixels(train_pixels[i],ids[i],view,config['augmentation_seed'],config['padding'])
                        for i in range(start,min(start+config['feature_batch'],len(ids)))])
                    images=transformed.to(device,dtype=torch.float32)/255
                    images=(images-base['image_mean'].to(device)[None,:,None,None])/base['image_std'].to(device)[None,:,None,None]
                    z=cnn.features(images)
                    x=torch.cat([(z-base['feature_mean'].to(device))/base['feature_std'].to(device),root_scores(cnn.head(z))],1)
                    output.append(x.cpu())
                    progress(stage='extracting_augmented_features',dataset=dataset,view=view,images_done=start+len(images),images_total=len(ids))
            features=torch.cat(output)
        save_checkpoint(destination,{'signature':signature,'view':view,'x':features})
    assert all(file_hash(p)==identity['source'][name] for name,p in source_paths.items())
    write_json(folder/'provenance.json',{'identity':identity,'old_sources_unchanged':True,
        'augmentation':'reflection-padded random crop + horizontal flip, keyed by original sample ID and view',
        'normalization':'Original frozen-CNN training statistics; no validation/test statistics.',
        'views_are_not_independent_original_samples':True,
        'bias_variance_scope':'Conditional on the shared pretrained representation; sample counts refer to images given to correction trees.'})
    return load_bundle(folder,config)


def load_bundle(folder,config):
    base=load_checkpoint(folder/'canonical.pt')
    rows=[]
    for view in range(config['views']):
        saved=load_checkpoint(folder/f'train-view-{view}.pt')
        assert saved['signature']==base['signature'] and saved['view']==view
        assert len(saved['x'])==len(base['splits']['train']['y'])
        rows.append(saved['x'])
    base['train_views']=torch.stack(rows)
    return base


def training_rows(bundle,draws,augmentation):
    if augmentation not in ['none','repeat','augmented']:raise ValueError('Unknown augmentation arm.')
    views=(bundle['train_views'] if augmentation=='augmented' else
           bundle['train_views'][:1].expand_as(bundle['train_views']) if augmentation=='repeat' else bundle['train_views'][:1])
    return views[:,draws].flatten(0,1).contiguous(),bundle['splits']['train']['y'][draws].repeat(len(views))


def test_features(dataset,bundle,source):
    # Loading saved test features is confined to the explicit finalization stage.
    source_path=WORKSPACE/source/dataset/'features.pt'
    assert file_hash(source_path)==bundle['source_feature_sha256']
    saved=load_checkpoint(source_path)
    manifest=read_json(WORKSPACE/'neural_tree/data'/dataset/'manifest.json')
    data=saved['splits']['test']
    assert data['ids']==[r['id'] for r in manifest['splits']['test']]
    assert torch.equal(data['y'],torch.tensor([r['label'] for r in manifest['splits']['test']]))
    return data
