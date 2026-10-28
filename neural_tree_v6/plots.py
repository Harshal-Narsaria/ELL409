"""Saved-data figures; no interpolation or smoothing of measured capacities."""
import argparse
import csv
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DEBUG_LABEL=False


def save(fig,path):
    if DEBUG_LABEL:
        if fig._suptitle:fig._suptitle.set_text('PILOT / debug subset: '+fig._suptitle.get_text())
        else:fig.suptitle('PILOT / debug subset')
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def plot_all(run,output):
    global DEBUG_LABEL
    from analysis import summarize
    records,decompositions,ensembles=summarize(run,output)
    destination=output/'figures';destination.mkdir(parents=True,exist_ok=True)
    if not records:return
    definition=read_json(run/'definition.json');cfg=definition['config']
    DEBUG_LABEL=cfg.get('debug_train_limit') is not None
    for dataset in cfg['datasets']:
        reference=[r for r in records if r['dataset']==dataset and r['job']['sampler']=='full' and r['job']['noise']==0
            and r['job']['width']==cfg['depth_width'] and r['job']['depth']==cfg['capacity_depth']
            and r['job']['epochs']==cfg['capacity_epochs'] and r['job']['seed']==cfg['seed']]
        reference.sort(key=lambda r:['none','repeat','augmented'].index(r['job']['augmentation']))
        if reference:
            fig,ax=plt.subplots(figsize=(8,4));positions=np.arange(len(reference));step=.2
            for offset,(key,label,color) in enumerate([('train_selected_unique','Canonical training','#7799bb'),('validation','Validation','#315477'),('test','Test','#d46533')]):
                values=[100*r[key]['accuracy'] if key in r else np.nan for r in reference]
                ax.bar(positions+(offset-1)*step,values,width=step,label=label,color=color)
            ax.set_xticks(positions,[r['job']['augmentation'] for r in reference])
            ax.set(ylabel='Accuracy (%)',title=f'{dataset}: transformed vs repeated-image exposure control')
            ax.legend(fontsize=8);save(fig,destination/f'{dataset}-reference-comparison.png')
        # Width and depth are separate experiments. Keep seed-specific curves;
        # adaptive topology means an active-parameter x-axis need not be ordered.
        for kind in ['width','depth']:
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            plotted=False
            for arm,color in [('none','#3f6d9c'),('augmented','#d46533')]:
                for noise in cfg['noise_rates'] if kind=='width' else [0.]:
                    rows=[r for r in records if r['dataset']==dataset and r['job']['sampler']=='full'
                        and r['job']['augmentation']==arm and r['job']['noise']==noise and r['job']['seed']==cfg['seed']
                        and r['job']['epochs']==cfg['capacity_epochs'] and
                        (r['job']['depth']==cfg['capacity_depth'] and r['job']['width'] in cfg['widths'] if kind=='width'
                         else r['job']['width']==cfg['depth_width'] and r['job']['depth'] in cfg['depths'])]
                    rows.sort(key=lambda r:r['job'][kind])
                    if not rows:continue
                    plotted=True;x=[r['size']['total_parameters'] for r in rows]
                    label=f'{arm}, noise {noise:.0%}'
                    style='--' if noise else '-'
                    axes[0].plot(x,[100*(1-r['train_views']['accuracy']) for r in rows],marker='o',ls=style,color=color,label=label)
                    axes[1].plot(x,[100*(1-r['validation']['accuracy']) for r in rows],marker='o',ls=style,color=color,label=label)
                    first=next((r for r in rows if r['train_views']['errors']==0),None)
                    if first:
                        for ax in axes:ax.axvline(first['size']['total_parameters'],color=color,ls=':',alpha=.4)
            for ax,title in zip(axes,['Finite-view training error','Clean validation error']):
                ax.set(xlabel='Allocated parameters (shared CNN counted once)',ylabel='Error (%)',title=title)
                ax.grid(alpha=.2)
                if plotted:ax.legend(fontsize=8)
            fig.suptitle(f'{dataset}: fixed-root {kind} study (node epochs {cfg["capacity_epochs"]})')
            if plotted:save(fig,destination/f'{dataset}-{kind}-capacity.png')
            else:plt.close(fig)
        # Both sample-size and epoch effects use the same definition of Brier.
        for axis_name in ['samples','epochs']:
            fig,axes=plt.subplots(1,3,figsize=(13,3.8));plotted=False
            for arm,color in [('none','#3f6d9c'),('augmented','#d46533')]:
                constants=cfg['epochs'] if axis_name=='samples' else cfg['sample_fractions']
                for constant in constants:
                    rows=[r for r in decompositions if r['dataset']==dataset and r['augmentation']==arm
                        and (r['epochs_per_child']==constant if axis_name=='samples' else r['sample_fraction']==constant)]
                    rows.sort(key=lambda r:r['sample_fraction'] if axis_name=='samples' else r['epochs_per_child'])
                    if not rows:continue
                    plotted=True
                    x=[sum(r['sample_draws'])/len(r['sample_draws']) if axis_name=='samples' else r['epochs_per_child'] for r in rows]
                    for ax,key in zip(axes,['squared_bias_proxy','bootstrap_prediction_variance','mean_member_brier']):
                        ax.plot(x,[r[key] for r in rows],marker='o',label=f'{arm}, '+(f'E={constant}' if axis_name=='samples' else f'n={constant:.0%}'))
            for ax,title in zip(axes,['Squared-bias proxy','Prediction variance','Mean member Brier']):
                ax.set(title=title,xlabel='Original-image draws' if axis_name=='samples' else 'Epochs per child');ax.grid(alpha=.2)
                if plotted:ax.legend(fontsize=7)
            fig.suptitle(f'{dataset}: conditional bootstrap bias/variance; not Bayes bias')
            if plotted:save(fig,destination/f'{dataset}-bias-variance-{axis_name}.png')
            else:plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.5,4));plotted=False
        for arm in ['none','augmented']:
            rows=[r for r in ensembles if r['dataset']==dataset and r['augmentation']==arm and r['epochs_per_child']==max(cfg['epochs'])]
            rows.sort(key=lambda r:r['members'])
            if rows:
                plotted=True;ax.plot([r['members'] for r in rows],[100*r['validation_accuracy'] for r in rows],'o-',label=arm)
        ax.set(xlabel='Complete image classifiers voting',ylabel='Validation accuracy (%)',title=f'{dataset}: majority vote');ax.grid(alpha=.2)
        if plotted:ax.legend();save(fig,destination/f'{dataset}-ensemble-voting.png')
        else:plt.close(fig)
    for path in (output/'landscapes').glob('*/landscape.npz'):
        raw=np.load(path);coordinates=raw['coordinates'];fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for i,title in enumerate(['Training loss','Validation loss','Gate-decision flips']):
            values=np.log1p(raw['losses'][i]) if i<2 else raw['gate_flip']
            im=axes[i].contourf(coordinates,coordinates,values.T,levels=20)
            axes[i].scatter([0],[0],color='red',s=16);axes[i].set(title=title+(' (log1p)' if i<2 else ''),xlabel='Direction 1 scale',ylabel='Direction 2 scale')
            fig.colorbar(im,ax=axes[i])
        fig.suptitle(path.parent.name+': actual hard gates; two local slices, not global geometry')
        save(fig,destination/(path.parent.name+'-landscape.png'))
    fields=['dataset','augmentation','noise','sampler','sample_fraction','replicate','seed','width','depth','epochs','draws','unique_originals',
            'finite_views','total_parameters','active_parameters','training_view_accuracy','validation_accuracy','test_accuracy']
    with (output/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:
            j=r['job'];writer.writerow({**{k:j[k] for k in fields[:10]},'draws':r['sample_draws'],
                'unique_originals':r['unique_originals'],'finite_views':r['finite_training_views'],
                'total_parameters':r['size']['total_parameters'],'active_parameters':r['size']['active_total_parameters'],
                'training_view_accuracy':r['train_views']['accuracy'],'validation_accuracy':r['validation']['accuracy'],
                'test_accuracy':r.get('test',{}).get('accuracy','')})
    lines=['# Augmentation follow-up: completed evidence','',f'Completed saved model configurations: {len(records)}.',
        '','These figures use completed configurations only. Canonical images, augmented views and bootstrap image draws are counted separately.',
        '','## Evidence boundaries','',cfg['bias_variance_scope'],'',cfg['test_policy'],
        '','Training curves locate interpolation. Held-out curves determine whether double descent is present; it is not guaranteed.',
        '','Epochs in these plots mean passes through each child task. Frozen roots do not train. The depth-one epoch study has fixed child tasks and resumes them at common epoch budgets.',
        '','Loss slices perturb MLP weights, keep parent strengths fixed, and recompute hard gates. They cannot establish a smooth Hessian or a global minimum.',
        '','## Latest complete full-data models','', '| Dataset | Augmentation | Noise | Width | Depth | Train views % | Validation % | Test % |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in records:
        j=r['job']
        if j['sampler']=='full':
            test=f"{100*r['test']['accuracy']:.2f}" if 'test' in r else 'withheld'
            lines.append(f"| {r['dataset']} | {j['augmentation']} | {j['noise']:.0%} | {j['width']} | {j['depth']} | {100*r['train_views']['accuracy']:.2f} | {100*r['validation']['accuracy']:.2f} | {test} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'runs/augmentation-study')
    parser.add_argument('--output',type=Path,default=PROJECT/'results/augmentation-study')
    a=parser.parse_args();plot_all(a.run_dir,a.output)
