"""Wait for the live worker, run the lower-width bracket, then refresh artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
RUN = PROJECT / 'runs/followup-chain'
RUN.mkdir(parents=True, exist_ok=True)

def status(stage, **extra):
    path = RUN / 'progress.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'pid': os.getpid(), 'stage': stage,
        'updated_unix': time.time(), **extra}, indent=2))
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(min(.05 * (attempt + 1), .5))

def read_progress(path):
    # Windows may briefly deny access while the writer replaces the file.
    # Retry on the next poll rather than abandoning the experiment queue.
    try:
        return json.loads(path.read_text())
    except (PermissionError, FileNotFoundError, json.JSONDecodeError):
        return None

def invoke(*args):
    subprocess.run([sys.executable, '-B', *map(str,args)], cwd=ROOT, check=True)

def main():
    sys.path.insert(0, str(PROJECT))
    import noise_study as study
    with study.study_lock(RUN / 'runner.lock'):
        status('waiting_for_main_worker')
        while True:
            p = PROJECT / 'runs/noise-followup/progress.json'
            progress = read_progress(p)
            if progress is None:
                status('waiting_for_progress_read')
                time.sleep(2)
                continue
            if progress.get('status') == 'complete': break
            try:
                with study.study_lock(PROJECT / 'runs/noise-followup/runner.lock'): pass
            except RuntimeError:
                status('waiting_for_main_worker',
                    main_completed_models=progress.get('completed_models'),
                    main_dataset=progress.get('dataset'))
                time.sleep(30)
                continue
            status('resuming_main_worker')
            invoke(PROJECT / 'noise_followup.py')
        status('lower_width_cats_bracket')
        invoke(PROJECT / 'noise_followup.py', '--config', PROJECT / 'cats-bracket-config.json',
            '--run-dir', PROJECT / 'runs/cats-bracket', '--output', PROJECT / 'results/cats-bracket')
        status('refreshing_report_and_package')
        invoke(ROOT / 'output/final_submission/prepare_noise_submission.py')
        invoke(ROOT / 'output/github-upload/ELL409-final-with-noise/tools/build_report.py')
        invoke(ROOT / 'output/final_submission/audit_noise_package.py')
        status('complete_visual_review_pending', models=48,
            note='All training and automated checks finished. The regenerated PDF needs a fresh visual review.')

if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
