"""Self-contained image deployments sharing one CNN across voting members."""
from common import *
from analysis import majority_vote


class ImageEnsemble(torch.nn.Module):
    def __init__(self,cnn,forests,image_mean,image_std,feature_mean,feature_std):
        super().__init__();self.cnn=cnn;self.members=torch.nn.ModuleList(forests)
        self.classes=forests[0].classes
        for p in self.parameters():p.requires_grad_(False)
        for name,value in [('image_mean',image_mean),('image_std',image_std),('feature_mean',feature_mean),('feature_std',feature_std)]:
            self.register_buffer(name,value.clone())

    def features(self,images):
        images=(images-self.image_mean[None,:,None,None])/self.image_std[None,:,None,None]
        z=self.cnn.features(images)
        return torch.cat([(z-self.feature_mean)/self.feature_std,root_scores(self.cnn.head(z))],1)

    def forward(self,images):
        x=self.features(images)
        predictions=torch.stack([member.predict(x) for member in self.members])
        return majority_vote(predictions,self.classes)[1]

    def predict(self,images):return self(images).argmax(1)

    def mean_probabilities(self,images):
        x=self.features(images)
        return torch.stack([probabilities(member(x),self.classes) for member in self.members]).mean(0)


def export_model(bundle,saved_members,path,metadata):
    save_checkpoint(path,{'format':'ELL409-augmentation-majority-v1','resolution':32,'classes':bundle['classes'],
        'cnn_model':bundle['cnn_model'],'members':[{'spec':m['spec'],'model':m['model']} for m in saved_members],
        **{k:bundle[k] for k in ['image_mean','image_std','feature_mean','feature_std']},
        'metadata':metadata,'voting':'One final class per member; maximum vote count, ties smallest class index.'})


def load_deployment(path,device='cpu'):
    saved=load_checkpoint(path)
    if saved['format']!='ELL409-augmentation-majority-v1':raise ValueError('Unsupported deployment format.')
    cnn=SharedCNN(saved['classes']);cnn.load_state_dict(saved['cnn_model'])
    forests=[]
    for member in saved['members']:
        forest=FeatureForest.from_spec(member['spec']);forest.load_state_dict(member['model']);forests.append(forest)
    model=ImageEnsemble(cnn,forests,saved['image_mean'],saved['image_std'],saved['feature_mean'],saved['feature_std'])
    return model.to(device).eval(),saved


def export_ensembles(run,output,bundles,config,primary):
    # Export whenever enough predeclared members finish, including partial runs.
    for dataset in config['datasets']:
        for arm in ['none','augmented']:
            jobs=[j for j in primary if j['dataset']==dataset and j['augmentation']==arm and j['sampler']=='bootstrap'
                and j['sample_fraction']==1 and j['epochs']==max(config['epochs'])]
            jobs.sort(key=lambda j:j['replicate'])
            from run import model_folder
            complete=[j for j in jobs if (model_folder(run,j)/'metrics.json').exists()]
            for count in [3,5]:
                if len(complete)<count:continue
                members=[load_checkpoint(model_folder(run,j)/'model.pt') for j in complete[:count]]
                export_model(bundles[dataset],members,output/'models'/f'{dataset}-{arm}-vote-{count}.pt',
                    {'jobs':complete[:count],'selection':'Predeclared members in replicate order, no held-out selection.',
                     'debug_subset':bundles[dataset]['debug_subset']})


if __name__=='__main__':
    import argparse
    from PIL import Image
    parser=argparse.ArgumentParser(description='Predict an image with one saved tree or voting ensemble.')
    parser.add_argument('model',type=Path);parser.add_argument('image',type=Path)
    a=parser.parse_args();model,metadata=load_deployment(a.model)
    pixels=np.asarray(Image.open(a.image).convert('RGB').resize((32,32),Image.Resampling.BILINEAR)).copy()
    x=torch.from_numpy(pixels).permute(2,0,1)[None].float()/255
    with torch.no_grad():votes=model(x)[0];label=int(votes.argmax())
    names=['cat','dog'] if model.classes==2 else ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print({'prediction':names[label],'votes':votes.tolist(),'members':len(model.members)})
