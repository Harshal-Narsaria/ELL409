from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queue_random_cnn import ordered_runs
from v2_tree.common import write_json


def test_queue_resumes_previous_before_new_study(tmp_path):
    calls=[]
    ordered_runs(tmp_path,lambda:calls.append('previous'),lambda:calls.append('random'))
    assert calls==['previous','random']


def test_queue_skips_completed_previous_study(tmp_path):
    write_json(tmp_path/'progress.json',{'status':'complete'})
    calls=[]
    ordered_runs(tmp_path,lambda:calls.append('previous'),lambda:calls.append('random'))
    assert calls==['random']
