"""Finish the active prototype, prioritize Version 3, then run fixed-root Version 2."""
from pathlib import Path
import os
import sys
import time

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent/'neural_tree_v2'))
from v2_tree.common import read_json, study_lock, write_json


def ordered_studies(priority_run, deferred_run):
    priority_run()
    deferred_run()


def main():
    run_dir = PROJECT/'runs/cnn80-depth'
    deferred = PROJECT.parent/'neural_tree_v2/runs/tree-budget-v2'
    prototype = PROJECT.parent/'neural_tree_v2/runs/image-tree-v2'
    with study_lock(run_dir/'queue.lock'), study_lock(deferred/'queue.lock'):
        write_json(run_dir/'queue-order.json', {'order':['Version 2 active prototype','Version 3 CNN-at-80%','Version 2 fixed-root'],
            'reason':'User requested prioritizing the new CNN-at-80% study on 2026-10-09.', 'updated_unix':time.time(),
            'preserves_completed_results':True,'experiments_and_seeds_unchanged':True})
        write_json(run_dir/'progress.json', {'status':'queued','pid':os.getpid(), 'updated_unix':time.time(),
            'waiting_for':['Version 2 active prototype'],'uses_test_data':True})
        write_json(deferred/'progress.json', {'status':'queued','pid':os.getpid(), 'updated_unix':time.time(),
            'waiting_for':'Version 3 CNN-at-80% study','uses_test_data':False})
        print('Priority updated: active prototype, then CNN-at-80% study, then fixed-root study.', flush=True)
        while True:
            try:
                with study_lock(prototype/'runner.lock'), study_lock(deferred/'runner.lock'):
                    pass
            except RuntimeError:
                time.sleep(15)
                continue
            break
        try:
            from study import run
            from v2_tree.study import study_main
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; resume when GPU is available.')
            device = torch.device('cuda')
            ordered_studies(
                lambda:run(PROJECT/'config.json',run_dir,PROJECT/'results/cnn80-depth',device),
                lambda:study_main(PROJECT.parent/'neural_tree',deferred,PROJECT.parent/'neural_tree_v2/results/tree-budget-v2',
                                  PROJECT.parent/'neural_tree_v2/configs/tree-budget-study.json',device,stage='all'))
        except Exception as exc:
            # Do not overwrite a completed Version 3 status if the deferred study failed.
            if not (run_dir/'progress.json').exists() or read_json(run_dir/'progress.json').get('status')!='complete':
                write_json(run_dir/'progress.json',{'status':'failed','error':repr(exc),'updated_unix':time.time()})
            raise


if __name__=='__main__': main()
