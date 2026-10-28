"""Finish the width-32 follow-up, then start the frozen random-CNN study."""
from pathlib import Path
import os
import sys
import time

PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT.parent/'neural_tree_v2'))
from v2_tree.common import read_json,study_lock,write_json


def previous_complete(previous):
    path=Path(previous)/'progress.json'
    return path.exists() and read_json(path).get('status')=='complete'


def ordered_runs(previous,previous_run,next_run):
    while True:
        try:
            with study_lock(Path(previous)/'runner.lock'): pass
        except RuntimeError:
            time.sleep(15)
            continue
        break
    if not previous_complete(previous): previous_run()
    next_run()


def main():
    previous=PROJECT.parent/'neural_tree_v3/runs/width32-followup'
    run_dir=PROJECT/'runs/random-cnn'
    with study_lock(run_dir/'queue.lock'):
        write_json(run_dir/'progress.json',{'status':'queued','pid':os.getpid(),'updated_unix':time.time(),
            'waiting_for':'Width-32 CIFAR-10 follow-up','order':['width32-followup','frozen random CNN'],
            'resume_command':'..\\.venv\\Scripts\\python.exe -B -u queue_random_cnn.py'})
        print('Queued frozen random-CNN study after the width-32 follow-up.',flush=True)
        def previous_run():
            sys.path.insert(0,str(PROJECT.parent/'neural_tree_v3'))
            from width32_followup import run
            import torch
            run(PROJECT.parent/'neural_tree_v3/width32-followup.json',previous,
                PROJECT.parent/'neural_tree_v3/results/width32-followup/cifar10',torch.device('cuda'))
        def next_run():
            from random_cnn_study import run
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; resume with a GPU available.')
            run(PROJECT/'config.json',run_dir,PROJECT/'results/random-cnn',torch.device('cuda'))
        try: ordered_runs(previous,previous_run,next_run)
        except Exception as exc:
            write_json(run_dir/'progress.json',{'status':'failed','updated_unix':time.time(),'error':repr(exc)})
            raise


if __name__=='__main__': main()
