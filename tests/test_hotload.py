"""CPU proofs: both ranks agree on hashes, then exactly those bytes run in place."""
import hashlib
import importlib
import os
import sys

import pytest

from tp_side_jobs.hotload import hotload, local_verdict

ORDER = ('stage',)


@pytest.fixture
def pkg(tmp_path):
    root = tmp_path / 'hotpkg'
    root.mkdir()
    (root / '__init__.py').write_text('')
    (root / 'stage.py').write_text('VALUE = 1\n')
    sys.path.insert(0, str(tmp_path))
    module = importlib.import_module('hotpkg.stage')
    yield root, module
    sys.path.remove(str(tmp_path))
    for name in [n for n in sys.modules if n.startswith('hotpkg')]:
        del sys.modules[name]


def staged(root, body):
    (root / 'stage.py').write_text(body)
    return {'stage.py': hashlib.sha256(body.encode()).hexdigest()}


def agree(name, value):
    return [value, value]


def test_reloads_in_place_when_both_ranks_agree(pkg):
    root, module = pkg
    expected = staged(root, 'VALUE = 2\n')
    report = hotload('hotpkg', root, expected, agree, order=ORDER)
    assert report == {'hotload': 'applied', 'reloaded': ['stage'], 'serving_restart': False}
    assert module.VALUE == 2 and sys.modules['hotpkg.stage'] is module


def test_refuses_on_peer_mismatch(pkg):
    root, module = pkg
    expected = staged(root, 'VALUE = 2\n')
    peer = {'ok': True, 'problems': [], 'hashes': {'stage.py': '0' * 64}}
    assert hotload('hotpkg', root, expected, lambda n, v: [v, peer], order=ORDER)['hotload'] == 'refused'
    assert module.VALUE == 1


def test_refuses_restart_only_and_unknown_files(pkg):
    root, _ = pkg
    verdict = local_verdict(root, {'bootstrap.py': '0' * 64, 'other.py': '0' * 64}, order=ORDER)
    assert not verdict['ok'] and len(verdict['problems']) == 2


def test_executes_new_bytes_even_when_size_and_mtime_match(pkg):
    root, module = pkg
    path = root / 'stage.py'
    stat = path.stat()
    expected = staged(root, 'VALUE = 9\n')       # same length as the original
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same mtime: a cached .pyc would look fresh
    hotload('hotpkg', root, expected, agree, order=ORDER)
    assert module.VALUE == 9
