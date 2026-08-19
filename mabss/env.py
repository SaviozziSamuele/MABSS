"""Walk-forward CMAB simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from .policies import make_policy
from .rewards import reward_function


@dataclass
class BanditRunResult:
    """
    Replaces the original `WalkForwardEnvCMAB.run()`'s 10-key dict, which mixed
    separator conventions (`'total return'` vs `'arm_selected'` vs
    `'best arms greedy'`) -- easy to typo, impossible to autocomplete.
    `to_legacy_dict()` reproduces the exact original keys/shapes for anything
    (the plotting functions, cached-artifact consumers) that still expects that
    shape.
    """

    total_return: np.ndarray
    arm_selected: np.ndarray
    best_arms: list
    total_reward_greedy: np.ndarray
    arm_selected_greedy: np.ndarray
    best_arms_greedy: list
    total_reward_average: np.ndarray
    weighted_predictions: np.ndarray
    mean_predictions_reward: np.ndarray
    control: list = field(default_factory=list)

    def to_legacy_dict(self) -> dict:
        return {
            "total return": self.total_return,
            "arm_selected": self.arm_selected,
            "best_arms": self.best_arms,
            "total reward greedy": self.total_reward_greedy,
            "arm selected greedy": self.arm_selected_greedy,
            "best arms greedy": self.best_arms_greedy,
            "total reward average": self.total_reward_average,
            "weighted predictions": self.weighted_predictions,
            "mean predictions reward": self.mean_predictions_reward,
            "control": self.control,
        }


class WalkForwardEnv:
    """
    Simulates a CMAB agent trading walk-forward over a price/prediction series.

    Ported from MABSS_utility.py's `WalkForwardEnvCMAB` (:752-873). Two bug fixes
    applied here, both discussed in the plan (Phase 4):

    1. **Stale `arm_probs` in the test loop** (originally `:858`). The original
       test loop computed `probs = self.policy.select_arm(x_test[t])` fresh each
       step, but then sampled the *stochastic* arm with
       `np.random.choice(..., p=arm_probs)` -- `arm_probs`, not `probs` -- where
       `arm_probs` was left over from the very last iteration of the *training*
       loop above it. For LinUCB/Thompson (one-hot `select_arm` output) this
       froze the stochastic arm to a single constant value for an entire test
       window; for Softmax it merely used a stale-but-nondegenerate
       distribution. Confirmed on the full corpus (35 bandit directories, all
       seeds): every UCB/Thompson directory's stochastic `best_arms` track has
       exactly 1 unique value per window; every Softmax directory doesn't (see
       tests/goldens/results_structure.json). **This does not affect the greedy
       track** (`best arms greedy = argmax(probs)`, computed fresh each step) --
       and per Manuscript.tex:197, greedy is the only track the paper reports.
       So this fix corrupts no published number; it only fixes a diagnostic
       array that was already wrong in every cached `results.pkl`/
       `multirun_results.pkl`.

    2. **Short-series guard** (originally `:831`). If `len_window >= len(x)`, the
       original loop body never executed and `run()` silently returned a dict of
       empty/all-zero arrays with no indication anything was wrong. This now
       raises a clear `ValueError` instead.

    Also: `policy_type` dispatch is case-insensitive via `mabss.policies.make_policy`
    (fixes `:797`, where `'Thompson'` raised while `'UCB'` didn't), and reward
    values passed to `calc_reward` for scalar-context call sites are squeezed to
    plain floats rather than left as reward_function's native (1,1) arrays (fixes
    the `'total reward average'`/`'mean predictions reward'` shape inconsistency
    noted in the plan -- `'mean predictions reward'` was a raw list of (1,1)
    arrays in the original, `'total reward average'` an ndarray of them).
    """

    def __init__(
        self,
        context,
        X_data,
        Y_data,
        rewards,
        reward_metric: str,
        k: int,
        d: int,
        config: dict,
        window_size: int = 252,
        policy_type: str = "softmax",
        incremental: bool = False,
        shuffle: bool = False,
        reset: bool = False,
    ):
        alpha_defaults = {"softmax": 0.01, "ucb": 1.0, "thompson": 0.1}
        alpha = config.get("ALPHA", alpha_defaults.get(policy_type.lower(), 1.0))
        gamma = config.get("GAMMA", None)
        self.policy_type = policy_type.lower()
        self.policy = make_policy(policy_type, k, d, alpha, gamma)

        self.x = context
        self.r = rewards
        self.metric = reward_metric
        self.X_data = X_data
        self.Y_data = Y_data
        self.w = window_size
        self.incremental = incremental
        self.shuffle = shuffle
        self.reset = reset

    def calc_reward(self, predictions, targets) -> float:
        """Same formula as the original (`reward_function`), but squeezed to a
        plain float for the scalar-prediction call sites in `run()` below."""
        return float(np.asarray(reward_function(predictions, targets, metric=self.metric)).squeeze())

    def run(self, episodes: int) -> BanditRunResult:
        total_reward = []
        arm_selected = np.zeros(self.policy.K_arms)
        best_arms = []
        total_reward_greedy = []
        arm_selected_greedy = np.zeros_like(arm_selected)
        best_arms_greedy = []
        total_reward_average = []
        mean_pred_reward = []
        weighted_predictions = []
        control = []

        len_window = (len(self.x) % self.w) + 4 * self.w
        if len_window >= len(self.x):
            raise ValueError(
                f"WalkForwardEnv.run: no walk-forward windows fit "
                f"(len(context)={len(self.x)}, computed training-window length="
                f"{len_window}, window_size={self.w}). The series is too short "
                "for this configuration. (The original silently returned a dict "
                "of empty arrays here.)"
            )

        for i in tqdm(range(len_window, len(self.x), self.w)):
            if self.incremental:
                x_train = self.x[:i]
                r_train = self.r[:i]
            else:
                x_train = self.x[(i - len_window) : i]
                r_train = self.r[(i - len_window) : i]
            x_test = self.x[i : (i + self.w)]
            r_test = self.r[i : (i + self.w)]
            X_data_test = self.X_data[i : (i + self.w)]
            Y_data_test = self.Y_data[i : (i + self.w)]
            control += list(Y_data_test)

            if self.reset:
                self.policy.reset()

            for _ in range(episodes):
                index = list(range(len(x_train)))
                if self.shuffle:
                    np.random.shuffle(index)
                for t in index:
                    arm_probs = self.policy.select_arm(x_train[t])
                    arm_index = np.random.choice(range(self.policy.K_arms), p=arm_probs)
                    self.policy.reward_update(
                        x_train[t], probs=arm_probs, arm_selected=arm_index, reward=r_train[t][arm_index]
                    )

            for t in range(len(x_test)):
                probs = self.policy.select_arm(x_test[t])
                # FIX (bug :858): sample from the freshly computed `probs`, not
                # the training loop's stale `arm_probs`. See class docstring.
                arm_index = np.random.choice(range(self.policy.K_arms), p=probs)
                arm_index_greedy = np.argmax(probs)

                best_arms.append(arm_index)
                arm_selected[arm_index] += 1
                total_reward.append(r_test[t][arm_index])

                best_arms_greedy.append(arm_index_greedy)
                arm_selected_greedy[arm_index_greedy] += 1
                total_reward_greedy.append(r_test[t][arm_index_greedy])

                prediction = np.dot(probs, X_data_test[t])
                weighted_predictions.append(prediction)
                total_reward_average.append(self.calc_reward(prediction, Y_data_test[t]))
                mean_pred_reward.append(self.calc_reward(np.mean(X_data_test[t]), Y_data_test[t]))

        return BanditRunResult(
            total_return=np.array(total_reward),
            arm_selected=arm_selected,
            best_arms=best_arms,
            total_reward_greedy=np.array(total_reward_greedy),
            arm_selected_greedy=arm_selected_greedy,
            best_arms_greedy=best_arms_greedy,
            total_reward_average=np.array(total_reward_average),
            weighted_predictions=np.array(weighted_predictions),
            mean_predictions_reward=np.array(mean_pred_reward),
            control=control,
        )
