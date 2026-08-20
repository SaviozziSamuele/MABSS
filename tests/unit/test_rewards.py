"""Unit tests for mabss.rewards.build_rewards_and_contexts, including the
optional shared market-risk context feature added for the context-vector
ablation (tools/build_context_ablation.py)."""

import numpy as np
import pandas as pd

from mabss.rewards import build_rewards_and_contexts


def _synthetic_pct_change_preds(T=100, n_arms=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    data = {"target": rng.normal(0.0005, 0.01, T)}
    model_cols = [f"model_{a}" for a in range(n_arms)]
    for col in model_cols:
        data[col] = rng.normal(0.0005, 0.01, T)
    df = pd.DataFrame(data, index=idx)
    df["mean"] = df[model_cols].mean(axis=1)
    return df, model_cols


def test_build_rewards_and_contexts_default_unchanged():
    """The new optional parameters must be fully inert unless explicitly
    requested -- proves the context-ablation flag can never affect the
    30 published-Table-1 configs, none of which pass it."""
    pct_change_preds, _ = _synthetic_pct_change_preds()
    bandit_embedding, n_arms = 15, 3

    rewards_df_a, contexts_a, rewards_mat_a = build_rewards_and_contexts(
        pct_change_preds, "l1", bandit_embedding, n_arms
    )
    rewards_df_b, contexts_b, rewards_mat_b = build_rewards_and_contexts(
        pct_change_preds, "l1", bandit_embedding, n_arms, include_market_risk_feature=False
    )

    pd.testing.assert_frame_equal(rewards_df_a, rewards_df_b)
    np.testing.assert_array_equal(contexts_a, contexts_b)
    np.testing.assert_array_equal(rewards_mat_a, rewards_mat_b)
    assert contexts_a.shape[-1] == bandit_embedding


def test_include_market_risk_feature_adds_one_dimension():
    pct_change_preds, _ = _synthetic_pct_change_preds()
    bandit_embedding, n_arms = 15, 3

    rewards_df, contexts, rewards_mat = build_rewards_and_contexts(
        pct_change_preds, "l1", bandit_embedding, n_arms, include_market_risk_feature=True, risk_feature_window=20
    )

    assert contexts.shape[-1] == bandit_embedding + 1
    assert contexts.shape[:2] == (rewards_mat.shape[0], n_arms)
    # No NaNs leaked through from the rolling-std warm-up period.
    assert not np.isnan(contexts).any()


def test_market_risk_feature_is_identical_across_arms_at_each_timestep():
    """The appended feature is a SHARED (non-arm-specific) market signal -- at
    any given timestep, every arm's extra context dimension must carry the
    exact same value, since it's the same rolling realized volatility of the
    single underlying target series."""
    pct_change_preds, _ = _synthetic_pct_change_preds(n_arms=4)
    bandit_embedding, n_arms = 15, 4

    _, contexts, _ = build_rewards_and_contexts(
        pct_change_preds, "l1", bandit_embedding, n_arms, include_market_risk_feature=True
    )

    extra_feature = contexts[:, :, -1]  # shape (T_ctx, K)
    for k in range(1, n_arms):
        np.testing.assert_array_equal(extra_feature[:, 0], extra_feature[:, k])


def test_market_risk_feature_reflects_realized_volatility_shape():
    """Sanity check on the feature's actual content: a series with a clearly
    higher-volatility second half should show a higher rolling-std feature
    value there than in the calmer first half."""
    T = 200
    idx = pd.bdate_range("2020-01-01", periods=T)
    rng = np.random.default_rng(2)
    calm = rng.normal(0.0, 0.001, T // 2)
    volatile = rng.normal(0.0, 0.05, T - T // 2)
    target = np.concatenate([calm, volatile])
    n_arms = 2
    data = {"target": target}
    model_cols = [f"model_{a}" for a in range(n_arms)]
    for col in model_cols:
        data[col] = target + rng.normal(0.0, 0.001, T)
    df = pd.DataFrame(data, index=idx)
    df["mean"] = df[model_cols].mean(axis=1)

    bandit_embedding = 15
    _, contexts, _ = build_rewards_and_contexts(
        df, "l1", bandit_embedding, n_arms, include_market_risk_feature=True, risk_feature_window=20
    )
    feature = contexts[:, 0, -1]
    early_mean = feature[: len(feature) // 4].mean()
    late_mean = feature[-len(feature) // 4 :].mean()
    assert late_mean > early_mean
