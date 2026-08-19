"""Walk-forward training of the predictor pool (MLP/RNN/CNN arms)."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .data import window_time_series
from .models import create_model


@dataclass(frozen=True)
class WalkForwardPlan:
    """
    Pure index arithmetic for the walk-forward training loop, split out of
    `walk_forward_training` specifically so the date-index bug (originally
    MABSS_utility.py:182) is unit-testable in milliseconds, with no Keras/model
    training involved.

    Terminology: "Xdata-space" means positions into `Xdata`/`Ydata` (the output of
    `window_time_series(data, embedding_len)`, length `n_samples = len(data) -
    embedding_len`); Ydata[i] == data.iloc[i + embedding_len]. "data-space" means
    positions into the original price series.

    The original bug: `walk_forward_training` built `preds_df`'s index as
    `data.iloc[-predictionstarget.shape[1]:].index` -- a TAIL slice of `data`. The
    walk-forward loop's window schedule tiles Xdata-space contiguously starting at
    `trainlen` (verified below), but stops as soon as
    `window_size*n_windows > n_samples - trainlen`, discarding a trailing
    remainder of 0..window_size-1 bars. A tail slice of `data` therefore
    systematically shifts every predicted row's date forward by exactly that
    remainder -- measured on the real corpus (see the plan) as SPY +22, BTC-USD
    +152, EURUSD=X +223, GC=F +93, TLT +133 bars.

    The correct index is POSITIONAL: predictions start at Xdata-space position
    `trainlen` (data-space position `trainlen + embedding_len`) and run for
    exactly `n_windows * window_size` bars -- windows are contiguous by
    construction (each window's test_end equals the next window's test_start,
    since the loop steps by exactly `window_size`).
    """

    n_samples: int  # len(Xdata) == len(data) - embedding_len
    trainlen: int
    window_size: int

    @property
    def window_starts(self) -> list[int]:
        """The `window` loop variable's values, in Xdata-space (mirrors the
        original `range(trainlen - 1, n_samples - window_size, window_size)`)."""
        return list(range(self.trainlen - 1, self.n_samples - self.window_size, self.window_size))

    @property
    def n_windows(self) -> int:
        return len(self.window_starts)

    @property
    def n_predictions(self) -> int:
        """Total predicted bars T = n_windows * window_size."""
        return self.n_windows * self.window_size

    @property
    def first_test_start(self) -> int:
        """Xdata-space position of the first test bar (== trainlen, given the
        first window == trainlen - 1, so test_start = window + 1 == trainlen)."""
        if self.n_windows == 0:
            raise ValueError(
                f"No walk-forward windows fit: n_samples={self.n_samples}, "
                f"trainlen={self.trainlen}, window_size={self.window_size}. "
                "The series is too short for this configuration."
            )
        return self.window_starts[0] + 1

    @property
    def n_unused_tail_bars(self) -> int:
        """Bars at the end of the series the walk-forward loop never scores,
        because the schedule stops once a full window no longer fits."""
        return self.n_samples - (self.first_test_start + self.n_predictions)

    def data_index_slice(self, embedding_len: int) -> slice:
        """The correct positional slice into the ORIGINAL price series' index
        (data-space, not Xdata-space) for the predictions this plan produces."""
        start = self.first_test_start + embedding_len
        return slice(start, start + self.n_predictions)

    def window_slices(self):
        """Yields (train_start, train_end, test_start, test_end) in Xdata-space,
        for each walk-forward window, matching the original loop body exactly."""
        for window in self.window_starts:
            train_start = window - self.trainlen + 1
            train_end = window + 1
            test_start = window + 1
            test_end = window + 1 + self.window_size
            yield train_start, train_end, test_start, test_end


def walk_forward_training(data: pd.Series, embedding_len: int, window_size: int, config: dict, epochs: int = 10):
    """
    Trains the full predictor pool (MLP/RNN/CNN, per CONFIG flags) walk-forward
    over `data`, and returns a DataFrame of out-of-sample predictions per arm.

    Ported from MABSS_utility.py:110-186. Changes vs. the original, all discussed
    in the plan:

      - **Correct date index** (Phase 4 item 5, ":182"): uses `WalkForwardPlan`
        above instead of a tail slice. `preds_df.attrs["n_unused_tail_bars"]`
        records how many trailing bars were never scored (0 means the series
        length was an exact fit); a warning is emitted when it's nonzero, since
        the original silently discarded up to `window_size - 1` bars per run
        with no record of it at all.
      - **K.clear_session() moved inside the arm loop** (Phase 2/4 item 9): the
        original called it once per walk-forward window, after all ~50-60 arms
        for that window were already built and trained, letting Keras/TF backend
        state for every arm in a window accumulate before being cleared. Moving
        it per-arm flattens that memory ramp. Coupled consequence (per the plan):
        this also resets Keras's global RNG state between arms, so trained
        weights differ from a run where clear_session() was called once per
        window -- this is a second, independent reason (alongside the MLP ReLU
        change) that regenerating experiments/ is required, not merely a
        preference.
      - **Dead `testtarget_scaled` computation removed** -- it was computed and
        never used in the original.
      - **Guards added**: empty `arms_list` (all of MLP/RNN/CNN False) and a
        series too short for even one walk-forward window now raise a clear
        `ValueError` instead of silently producing an all-NaN 'mean' column or a
        bare `ValueError: need at least one array to concatenate` respectively.
      - **tqdm.auto instead of tqdm.notebook** (Phase 2, ":4") -- the original
        import broke outside a live Jupyter/ipywidgets kernel.

    NOT changed here: the per-arm seeding scheme (`create_model(arch, seed=m,
    ...)` where `m` is the arm's index within its architecture, reused across
    every walk-forward window) -- see mabss.seeding.derive_arm_seed's docstring
    for why the "minimal delta" option keeps this as-is.
    """
    import keras.backend as K

    arms_list: list[tuple[str, int]] = []
    if config.get("MLP", False):
        arms_list += [("mlp", m) for m in range(config["N_SEEDS_PER_ARCH"])]
    if config.get("RNN", False):
        arms_list += [("rnn", m) for m in range(config["N_SEEDS_PER_ARCH"])]
    if config.get("CNN", False):
        arms_list += [("cnn", m) for m in range(config["N_SEEDS_PER_ARCH"])]

    if not arms_list:
        raise ValueError(
            "walk_forward_training: CONFIG has none of MLP/RNN/CNN enabled -- "
            "there is nothing to train. (The original code silently produced an "
            "all-NaN 'mean' column in this case.)"
        )

    n_models = len(arms_list)
    print(f"Generating {n_models} arms")

    Xdata, Ydata = window_time_series(data, embedding_len)
    trainlen = 4 * window_size
    n_samples = len(Xdata)

    plan = WalkForwardPlan(n_samples=n_samples, trainlen=trainlen, window_size=window_size)
    if plan.n_windows == 0:
        raise ValueError(
            f"walk_forward_training: series too short for even one walk-forward "
            f"window (n_samples={n_samples}, trainlen={trainlen}, "
            f"window_size={window_size}). (The original code raised a bare "
            "'ValueError: need at least one array to concatenate' later on.)"
        )
    if plan.n_unused_tail_bars:
        import warnings

        warnings.warn(
            f"walk_forward_training: {plan.n_unused_tail_bars} trailing bars of "
            "the series will not be scored by any walk-forward window (the "
            "schedule only advances in full window_size steps). This is normal, "
            "not a bug -- just make sure it's accounted for when reporting the "
            "evaluated date range.",
            stacklevel=2,
        )

    predictions = {f"model_{i}": [] for i in range(n_models)}
    predictionstarget = []

    for train_start, train_end, test_start, test_end in tqdm(list(plan.window_slices())):
        traindata = Xdata[train_start:train_end]
        traintarget = Ydata[train_start:train_end].reshape(-1, 1)
        testdata = Xdata[test_start:test_end]
        testtarget = Ydata[test_start:test_end].reshape(-1, 1)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        traindata_scaled = x_scaler.fit_transform(traindata)
        testdata_scaled = x_scaler.transform(testdata)
        traintarget_scaled = y_scaler.fit_transform(traintarget)

        predictionstarget.append(testtarget.reshape(1, -1))

        traindata_seq = traindata_scaled.reshape(traindata_scaled.shape + (1,))
        testdata_seq = testdata_scaled.reshape(testdata_scaled.shape + (1,))

        for i, (arch, seed) in enumerate(arms_list):
            model = create_model(arch, seed, embedding_len)

            if arch == "mlp":
                model.fit(traindata_scaled, traintarget_scaled, epochs=epochs, verbose=0)
                pred_scaled = model.predict(testdata_scaled, verbose=0)
            else:
                model.fit(traindata_seq, traintarget_scaled, epochs=epochs, verbose=0)
                pred_scaled = model.predict(testdata_seq, verbose=0)

            pred = y_scaler.inverse_transform(pred_scaled)
            predictions[f"model_{i}"].append(pred.reshape(1, -1))
            del model
            K.clear_session()
        gc.collect()

    for i in range(n_models):
        predictions[f"model_{i}"] = np.concatenate(predictions[f"model_{i}"], axis=1)
    predictionstarget = np.concatenate(predictionstarget, axis=1)

    index_slice = plan.data_index_slice(embedding_len)
    preds_df = pd.DataFrame(
        np.hstack([predictionstarget.T] + [predictions[f"model_{i}"].T for i in range(n_models)]),
        columns=["target"] + [f"model_{i}" for i in range(n_models)],
        index=data.iloc[index_slice].index,
    )
    preds_df["mean"] = preds_df[[f"model_{i}" for i in range(n_models)]].mean(axis=1)
    preds_df.attrs["n_unused_tail_bars"] = plan.n_unused_tail_bars

    return preds_df
