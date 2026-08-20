"""
MABSS: Dynamic Predictor Selection for Financial Time Series Using Contextual
Multi-Armed Bandits in a Reinforcement Learning Framework.

Note on import structure: the plan called for PEP 562 lazy `__getattr__` at
package scope specifically so that determinism-related environment variables
(TF_DETERMINISTIC_OPS etc., set by `mabss.seeding.set_global_seed`) could be
set before TensorFlow/Keras is ever imported. That constraint is satisfied here
more simply: `mabss.models.create_model` and `mabss.training.walk_forward_training`
import `keras`/`keras.backend` INSIDE the function body, not at module scope --
so `import mabss` (which does eagerly import every submodule below) never
triggers a TensorFlow import. Call `mabss.seeding.set_global_seed(...)` before
the first call to `create_model`/`walk_forward_training`, not before `import
mabss` itself.
"""

from . import metrics, plots, seeding, strategies
from .data import compute_returns_from_preds, load_price_series, window_time_series
from .env import BanditRunResult, WalkForwardEnv
from .experiments import (
    AmbiguousExperimentError,
    ExperimentNotFoundError,
    ExperimentStore,
    LegacyIndex,
    LegacyMatch,
    experiment_id,
    make_experiment_id,
)
from .models import create_model
from .policies import LinUCBPolicy, POLICY_REGISTRY, Policy, SoftmaxPolicy, ThompsonSamplingPolicy, make_policy
from .rewards import build_rewards_and_contexts, reward_function
from .runner import run_bandit_multi, run_bandit_seed
from .training import WalkForwardPlan, walk_forward_training

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "metrics",
    "seeding",
    "strategies",
    "plots",
    "run_bandit_seed",
    "run_bandit_multi",
    "load_price_series",
    "window_time_series",
    "compute_returns_from_preds",
    "create_model",
    "reward_function",
    "build_rewards_and_contexts",
    "Policy",
    "SoftmaxPolicy",
    "LinUCBPolicy",
    "ThompsonSamplingPolicy",
    "POLICY_REGISTRY",
    "make_policy",
    "WalkForwardEnv",
    "BanditRunResult",
    "WalkForwardPlan",
    "walk_forward_training",
    "ExperimentStore",
    "make_experiment_id",
    "experiment_id",
    "LegacyIndex",
    "LegacyMatch",
    "AmbiguousExperimentError",
    "ExperimentNotFoundError",
]
