import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('priority_queue',Path(__file__).resolve().parents[1]/'queue-study.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_priority_runs_before_deferred_study():
    events=[]
    module.ordered_studies(lambda:events.append('cnn80'),lambda:events.append('fixed_root'))
    assert events==['cnn80','fixed_root']


def test_priority_failure_stops_dispatching_new_studies():
    events=[]
    def fail():
        events.append('cnn80')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):
        module.ordered_studies(fail,lambda:events.append('fixed_root'))
    assert events==['cnn80']
