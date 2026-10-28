"""Evaluate every completed point under the predeclared time-limit fallback."""
import time
import torch
import root_gap as gap
s=gap.s


def main():
    amendment=s.read_json(gap.PROJECT/'root-gap-time-budget-amendment.json')
    assert time.time()>=amendment['primary_training_cutoff_unix']
    status=s.read_json(gap.RUN/'progress.json')
    assert status['status']=='stopped_at_user_time_limit',status
    with s.study_lock(gap.RUN/'runner.lock'):
        definition=s.read_json(gap.RUN/'definition.json')
        cfg=definition['config'];job=definition['job']
        folder=gap.RUN/'cifar10/noise-1500'
        complete=sorted(int(p.parent.name.split('-')[1]) for p in folder.glob('seed-409/width-*/metrics.json'))
        assert complete
        for width in complete:
            for c in range(10):
                checkpoint=s.load_checkpoint(folder/f'seed-409/width-{width}/bases/class-{c}/root.pt')
                assert checkpoint['epoch']==job['root_epochs']==1200
        audit=s.read_json(gap.RUN/'preserved-source-roots.json')
        assert all(s.file_hash(gap.Path(p))==v for p,v in audit.items())
        source,metadata=s.source_bundle('cifar10',definition['base'])
        bundle=s.noisy_bundle(source,.15,definition['base']['noise_seed'],folder)
        eval_config={**cfg,'widths':complete}
        s.finalize_test(folder,bundle,eval_config,torch.device('cuda'))
        for width in complete:
            path=folder/f'seed-409/width-{width}/metrics.json'
            record=s.read_json(path)
            record['test_status']='evaluated_after_predeclared_time_cutoff; incomplete_widths_excluded; reused_test_split_exploratory'
            s.write_json(path,record)
        s.export_results(gap.RUN/'cifar10',gap.OUTPUT/'cifar10',eval_config)
        status.update(status='complete_partial_time_limited',completed_models=len(complete),
                      completed_widths=complete,incomplete_widths=sorted(set(job['widths'])-set(complete)),
                      previous_roots_verified_unchanged=True,updated_unix=time.time(),
                      test_evaluated_by_predeclared_time_rule=True)
        s.write_json(gap.RUN/'progress.json',status)
        print(status)


if __name__=='__main__':main()
