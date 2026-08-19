"""
Regression tests for the walk-forward prediction date-index bug
(originally MABSS_utility.py:182), fixed by mabss.training.WalkForwardPlan.

The original code indexed `preds_df` with a TAIL slice of the price series
(`data.iloc[-predictionstarget.shape[1]:].index`), which is wrong whenever the
walk-forward loop's window schedule leaves a trailing remainder unscored (i.e.
whenever `(n_samples - trainlen)` is not an exact multiple of `window_size`).
The correct index is positional: predictions start at Xdata-space position
`trainlen` (data-space position `trainlen + embedding_len`) and run for exactly
`n_windows * window_size` bars.

Confirmed on the real corpus (see the plan): with trainlen=1008, window_size=252,
embedding_len=15, the true offset is 1023 = 1008 + 15, and measured shifts on
cached artifacts were SPY +22, BTC-USD +152, EURUSD=X +223, GC=F +93, TLT +133
bars -- i.e. every published prediction was date-stamped up to ~11 months later
than the bar it actually corresponds to.
"""

import pytest

from mabss.training import WalkForwardPlan


def test_first_test_start_equals_trainlen():
    """The first test window always starts at Xdata-space position == trainlen,
    since the loop's first `window` value is `trainlen - 1` and
    `test_start = window + 1`."""
    plan = WalkForwardPlan(n_samples=5528, trainlen=1008, window_size=252)
    assert plan.first_test_start == 1008


def test_data_index_slice_start_matches_measured_spy_offset():
    """Locks in the forensic finding from the plan: trainlen=1008 +
    embedding_len=15 == 1023, the offset that best-matched a fresh SPY download
    against the committed preds.csv (max abs error 3.68 under auto_adjust=True,
    vs. 45.3 for the next-best offset)."""
    plan = WalkForwardPlan(n_samples=5528, trainlen=1008, window_size=252)
    index_slice = plan.data_index_slice(embedding_len=15)
    assert index_slice.start == 1023


def test_windows_tile_contiguously():
    """Each window's test_end must equal the next window's test_start -- this is
    what makes a single positional slice valid for the whole prediction run
    (rather than needing per-window date stitching)."""
    plan = WalkForwardPlan(n_samples=5528, trainlen=1008, window_size=252)
    slices = list(plan.window_slices())
    assert len(slices) == plan.n_windows
    for (_, _, _, test_end_i), (_, _, test_start_next, _) in zip(slices, slices[1:]):
        assert test_end_i == test_start_next


def test_n_predictions_equals_n_windows_times_window_size():
    plan = WalkForwardPlan(n_samples=5528, trainlen=1008, window_size=252)
    assert plan.n_predictions == plan.n_windows * 252
    assert plan.n_predictions > 0


def test_n_unused_tail_bars_matches_hand_computed_case():
    """Constructs an n_samples where the tail remainder is known by
    construction: n_samples = trainlen + 2*window_size + 50 gives exactly 2
    windows and a 50-bar unscored remainder."""
    trainlen, window_size = 1008, 252
    n_samples = trainlen + 2 * window_size + 50
    plan = WalkForwardPlan(n_samples=n_samples, trainlen=trainlen, window_size=window_size)
    assert plan.n_windows == 2
    assert plan.n_predictions == 2 * window_size
    assert plan.n_unused_tail_bars == 50


def test_zero_unused_tail_bars_when_series_length_is_exact_fit():
    trainlen, window_size = 1008, 252
    n_samples = trainlen + 3 * window_size  # exact fit, no remainder
    plan = WalkForwardPlan(n_samples=n_samples, trainlen=trainlen, window_size=window_size)
    assert plan.n_unused_tail_bars == 0


def test_raises_when_no_window_fits():
    """The original silently returned an empty preds_df with no warning when the
    series was too short (later manifesting as a bare
    'ValueError: need at least one array to concatenate' deep in
    walk_forward_training). WalkForwardPlan raises immediately and clearly."""
    plan = WalkForwardPlan(n_samples=500, trainlen=1008, window_size=252)
    with pytest.raises(ValueError, match="No walk-forward windows fit"):
        _ = plan.first_test_start


def test_data_index_slice_length_equals_n_predictions():
    plan = WalkForwardPlan(n_samples=5528, trainlen=1008, window_size=252)
    index_slice = plan.data_index_slice(embedding_len=15)
    assert (index_slice.stop - index_slice.start) == plan.n_predictions
