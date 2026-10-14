"""Runs the two-process gloo demo end to end (CPU only, localhost)."""
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip('torch.distributed')
PATH = Path(__file__).resolve().parents[1] / 'examples' / 'two_rank_demo.py'


def demo():
    # Spawned ranks re-import the module by name, so it must be importable from sys.path.
    if str(PATH.parent) not in sys.path:
        sys.path.insert(0, str(PATH.parent))
    return importlib.import_module('two_rank_demo')


def test_both_ranks_run_the_job_in_lockstep_serve_during_it_and_hot_load():
    logs = demo().run()
    assert logs[0] == logs[1]                       # identical event order on both ranks
    kinds = [row[0] for row in logs[0]]
    assert kinds.count('job_step') == 3 and 'served_during_job' in kinds
    assert ('hotload', 'applied', 2) in logs[0]
    assert kinds[-1] == 'chat'                      # serving continues after the job


def test_message_mismatch_cancels_only_the_job():
    logs = demo().run(mismatch=True)
    for log in logs.values():
        kinds = [row[0] for row in log]
        assert 'cancelled' in kinds and 'job_step' not in kinds
        assert kinds[-1] == 'chat'
