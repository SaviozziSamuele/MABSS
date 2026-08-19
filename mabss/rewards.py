"""Reward computation and CMAB context/reward matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import window_time_series


def reward_function(
    predictions, targets, metric: str = "l1", prob_scale: float = 50.0, eps: float = 1e-6
):
    """
    predictions: array-like, shape (N, K) or (N,)  (predicted returns, one column per arm)
    targets:     array-like, shape (N,) or (N, 1)  (true returns)
    metric:      'l1', 'l2', or 'crossentropy'
    prob_scale:  scale factor for mapping returns -> probabilities (crossentropy)

    Ported from MABSS_utility.py:352-408, logic unchanged. Always returns a 2-D
    array (N, K) even for scalar/1-D input, matching the original -- callers that
    need a scalar back (e.g. mabss.env's per-step average/mean-prediction reward)
    are responsible for squeezing; see mabss.env.WalkForwardEnv.
    """
    pred_arr = predictions.values if hasattr(predictions, "values") else np.asarray(predictions)
    targ_arr = targets.values if hasattr(targets, "values") else np.asarray(targets)

    if pred_arr.ndim == 0:
        pred_arr = pred_arr[None]
    if targ_arr.ndim == 0:
        targ_arr = targ_arr[None]

    if pred_arr.ndim == 1:
        pred_arr = pred_arr[:, None]
    if targ_arr.ndim == 1:
        targ_arr = targ_arr[:, None]

    if pred_arr.shape[0] != targ_arr.shape[0]:
        raise ValueError(f"Incompatible first dimensions: {pred_arr.shape} vs {targ_arr.shape}")

    if metric == "l1":
        return -np.abs(pred_arr - targ_arr)

    elif metric == "l2":
        return -np.square(pred_arr - targ_arr)

    elif metric == "crossentropy":
        true_classes = (targ_arr > 0).astype(float)
        logits = prob_scale * pred_arr
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, eps, 1.0 - eps)
        ce_loss = -(true_classes * np.log(probs) + (1.0 - true_classes) * np.log(1.0 - probs))
        return -ce_loss

    else:
        raise ValueError(f"Invalid metric: {metric}")


def build_rewards_and_contexts(
    pct_change_preds: pd.DataFrame, metric: str, bandit_embedding: int, n_arms: int
):
    """
    Returns:
        rewards_df: DataFrame with columns ['target', 'model_0', ..., 'model_{K-1}', 'mean']
        contexts_matrix: np.ndarray, shape (T_ctx, K, bandit_embedding)
        rewards_matrix: np.ndarray, shape (T_ctx, K)

    Ported from MABSS_utility.py:410-449, logic unchanged. Requires a 'mean'
    column in `pct_change_preds` (added by the caller, e.g. the notebook's
    `pct_change_preds['mean'] = pct_change_preds[model_cols].mean(axis=1)`) --
    this is not created by `compute_returns_from_preds` itself; a missing 'mean'
    column raises a plain KeyError here, same as the original.

    Row j of contexts_matrix/rewards_matrix corresponds to pct_change_preds row
    j + bandit_embedding (via window_time_series). Contexts are windows of each
    arm's own *reward* series (not of its returns) -- deliberate, per the reward
    function feeding the bandit's context.
    """
    target = pct_change_preds["target"].values
    model_cols = [f"model_{n}" for n in range(n_arms)]
    preds_mat = pct_change_preds[model_cols].values

    rewards_mat = reward_function(preds_mat, target, metric=metric)

    rewards_df = pd.DataFrame(
        np.column_stack([target, rewards_mat]),
        index=pct_change_preds.index,
        columns=["target"] + model_cols,
    )
    rewards_df["mean"] = reward_function(pct_change_preds["mean"], target, metric=metric)

    context_list = []
    reward_list = []
    for col in model_cols:
        arm_contexts, arm_rewards = window_time_series(rewards_df[col], bandit_embedding)
        context_list.append(arm_contexts)
        reward_list.append(arm_rewards)

    contexts_matrix = np.stack(context_list, axis=1)
    rewards_matrix = np.stack(reward_list, axis=1)

    return rewards_df, contexts_matrix, rewards_matrix
