# SPDX-License-Identifier: Apache-2.0
"""Load new stage code into the live serving ranks without a restart.

Runs as the sealed `hotload` thin job: both ranks enter it at the same serving
message, hash the staged files, agree over the trainer channel, and only then
re-execute the hashed source of each listed module in place (same module object, so
hooks that look names up at call time see the new code). Serving weights,
the registration hook and the engine are untouched.

What this cannot do (still needs a paired restart, see BUILD_NOTES.md): change the
hook registration (`autograd_native._register_thinner`), the engine/tensorfold
sources, or the bodies of the closures `thin_job.attach` already installed.
"""
import hashlib
import sys
from pathlib import Path

# Defaults from the origin trainer. Pass your own `order` (leaves first, the module whose
# frame is running last) and `restart_only` (files whose objects are held by live hooks).
ORDER = ("fit", "scope", "answer", "thin_job")
# The hook registration module (`autograd_native.py` in the origin engine), the engine
# bootstrap, and the autograd/budget modules whose objects are held by live closures.
RESTART_ONLY = ("autograd_native.py", "thin_hooks.py", "bootstrap.py", "thin.py", "thin_native.py",
                "thin_budget.py", "settings.py")


def local_verdict(package_dir, expected, order=ORDER, restart_only=RESTART_ONLY):
    """Hash each staged file; refuse names that need a restart or are not known."""
    problems, hashes = [], {}
    for name, digest in sorted(expected.items()):
        if name in restart_only:
            problems.append(name + " can only change with a paired restart")
            continue
        if name[:-3] not in order or not name.endswith(".py"):
            problems.append(name + " is not a hot-loadable module")
            continue
        path = Path(package_dir) / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        hashes[name] = actual
        if actual != digest:
            problems.append(name + " staged hash differs")
    return {"ok": not problems, "problems": problems, "hashes": hashes}


def hotload(package, package_dir, expected, exchange, modules=None, order=ORDER, restart_only=RESTART_ONLY):
    """Verify on this rank, agree with the peer, then reload. Returns report fields."""
    mine = local_verdict(package_dir, expected, order, restart_only)
    both = exchange("hotload-verdict", mine)
    if not all(v["ok"] for v in both) or both[0]["hashes"] != both[-1]["hashes"]:
        return {"hotload": "refused", "verdicts": both}
    modules = sys.modules if modules is None else modules
    reloaded = []
    for short in order:
        if short + ".py" not in expected:
            continue
        name = package + "." + short
        if name in modules:
            # Execute exactly the bytes that were hashed, into the existing module
            # object; no bytecode cache can serve stale code.
            raw = (Path(package_dir) / (short + ".py")).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected[short + ".py"]:
                raise RuntimeError(short + ".py changed between verify and load")
            module = modules[name]
            exec(compile(raw, module.__file__, "exec"), module.__dict__)  # noqa: S102
            reloaded.append(short)
        # A module never imported yet loads fresh on first use; nothing to do.
    done = exchange("hotload-done", reloaded)
    if done[0] != done[-1]:
        raise RuntimeError("Ranks reloaded different modules: " + str(done))
    return {"hotload": "applied", "reloaded": reloaded, "serving_restart": False}
