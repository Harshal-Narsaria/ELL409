"""Run the predeclared two-width follow-up within the user's time limit."""
import os
from pathlib import Path
import time
import torch
import noise_followup as followup

PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / 'one-hour-config.json'
RUN = PROJECT / 'runs/one-hour-bracket'
OUTPUT = PROJECT / 'results/one-hour-bracket'


class TimeBudgetReached(Exception):
    pass


def main():
    job = followup.s.read_json(CONFIG)
    started = time.monotonic()
    deadline = started + job['wall_clock_budget_seconds']
    original_fit = followup.s.fit_node
    original_stage = followup.s.fit_stage

    def check_budget():
        if time.monotonic() >= deadline:
            raise TimeBudgetReached('48-minute wall-clock limit reached; epoch checkpoint retained.')

    def bounded_fit(model, images, indices, labels, path, local, identity, progress=None):
        check_budget()

        def saved_epoch(**fields):
            if progress:
                progress(**fields)
            check_budget()

        return original_fit(model, images, indices, labels, path, local, identity, saved_epoch)

    def bounded_stage(*args, **kwargs):
        check_budget()
        return original_stage(*args, **kwargs)

    followup.s.fit_node = bounded_fit
    followup.s.fit_stage = bounded_stage
    try:
        followup.run(CONFIG, RUN, OUTPUT, torch.device('cuda'))
    except TimeBudgetReached as exc:
        state = followup.s.read_json(RUN / 'progress.json')
        state.update(status='stopped_at_user_time_limit', error=None,
                     elapsed_seconds=time.monotonic()-started, reason=str(exc),
                     incomplete_runs_excluded=True, updated_unix=time.time())
        followup.s.write_json(RUN / 'progress.json', state)
        print(str(exc), flush=True)
    finally:
        followup.s.fit_node = original_fit
        followup.s.fit_stage = original_stage


if __name__ == '__main__':
    main()
