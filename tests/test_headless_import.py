"""
Regression test for the tqdm.notebook import bug (originally MABSS_utility.py:4,
`from tqdm.notebook import tqdm`).

`tqdm.notebook.tqdm` doesn't raise on import -- only on first instantiation,
when it tries to build an ipywidgets progress bar and finds no IPython kernel /
no ipywidgets installed (`ImportError: IProgress not found`). Since
MABSS_utility.py is imported by both the notebook AND (per the plan's Data
Availability finding) whatever headless driver produced the 2.3 GB corpus on
Toeplitz, this made the module's core training/bandit loops unusable outside a
live Jupyter kernel.

mabss.training and mabss.env both import `from tqdm.auto import tqdm` instead,
and MABSS_utility.py's compat-override section (bottom of the file) rebinds
`walk_forward_training`/`WalkForwardEnvCMAB` to the mabss versions -- so by the
time anything actually calls them, the notebook-only tqdm is never instantiated
(the module still contains the original tqdm.notebook-based function bodies as
dead, shadowed code -- see the compat section's docstring for why that cleanup
is deferred).

Runs in a real subprocess (not just this process) so it can't be fooled by
whatever happens to already be imported in the current pytest process.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_ipywidgets_is_not_installed_in_this_environment():
    """Sanity check that this test environment actually exercises the bug this
    module guards against -- if ipywidgets were installed, tqdm.notebook would
    silently work and this whole test file would be testing nothing."""
    result = subprocess.run(
        [sys.executable, "-c", "import ipywidgets"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "ipywidgets IS installed in this environment -- the headless-import "
        "regression this test guards against can't be observed here. This "
        "isn't a failure of the code under test, but it means this test file "
        "isn't actually exercising anything; investigate the test environment."
    )


def test_tqdm_notebook_would_fail_headless_confirming_the_bug_exists():
    """Documents that the ORIGINAL bug (`from tqdm.notebook import tqdm`,
    instantiated) really does raise in this headless environment -- i.e. that
    the fix below is fixing something real, not a phantom."""
    code = "from tqdm.notebook import tqdm; tqdm(range(3)).__enter__()"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "IProgress" in result.stderr or "ImportError" in result.stderr


def test_mabss_training_and_env_import_and_run_headless():
    """The actual fix: mabss.training / mabss.env must import and run a real
    (tiny) progress-bar-using loop in a bare subprocess with no IPython kernel
    and no ipywidgets -- exactly the environment the original code broke in."""
    code = """
import sys
sys.path.insert(0, r"{repo}")
from mabss.env import WalkForwardEnv
import numpy as np

rng = np.random.default_rng(0)
T, K, d = 100, 3, 2
contexts = rng.normal(size=(T, K, d))
rewards = rng.normal(size=(T, K))
X_data = rng.normal(size=(T, K))
Y_data = rng.normal(size=T)

env = WalkForwardEnv(
    context=contexts, X_data=X_data, Y_data=Y_data, rewards=rewards,
    reward_metric="l1", k=K, d=d, config={{"ALPHA": 1.0, "GAMMA": None}},
    window_size=10, policy_type="ucb", shuffle=False, reset=True,
)
result = env.run(episodes=1)
assert len(result.best_arms_greedy) > 0
print("HEADLESS_ENV_RUN_OK")
""".format(repo=str(REPO_ROOT))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "HEADLESS_ENV_RUN_OK" in result.stdout


def test_mabss_utility_star_import_works_headless():
    """The full compat shim, exercised in a bare subprocess: `from
    MABSS_utility import *` (the notebook's own import statement) must succeed,
    and the rebound WalkForwardEnvCMAB must be runnable, with no IPython kernel
    present."""
    code = """
import sys
sys.path.insert(0, r"{repo}")
from MABSS_utility import *
import numpy as np

rng = np.random.default_rng(0)
T, K, d = 100, 3, 2
contexts = rng.normal(size=(T, K, d))
rewards = rng.normal(size=(T, K))
X_data = rng.normal(size=(T, K))
Y_data = rng.normal(size=T)

env = WalkForwardEnvCMAB(
    context=contexts, X_data=X_data, Y_data=Y_data, rewards=rewards,
    reward_metric="l1", k=K, d=d, config={{"ALPHA": 1.0, "GAMMA": None}},
    window_size=10, policy_type="ucb", shuffle=False, reset=True,
)
result = env.run(episodes=1)
assert isinstance(result, dict)
assert "best arms greedy" in result
print("HEADLESS_STAR_IMPORT_RUN_OK")
""".format(repo=str(REPO_ROOT))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "HEADLESS_STAR_IMPORT_RUN_OK" in result.stdout
