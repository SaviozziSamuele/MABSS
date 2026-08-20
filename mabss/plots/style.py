"""Shared plot-saving helper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig, path, dpi: int = 300) -> None:
    """
    Saves `fig` to an EXPLICIT `path` and closes it.

    Replaces MABSS_utility.py's `save_current_plot` (:925-953), which derived
    the filename from the axes title (falling back to a caller-supplied
    `title_fallback` only when there was no title at all -- which never
    happened, since every caller sets one). Every title in the original module
    is static and ticker-free (e.g. `'Cumulative return of CMAB-derived
    strategies'`), so running the pipeline for SPY then BTC-USD into the same
    `save_dir` silently overwrote every prior figure -- confirmed live in the
    committed corpus: all 30 `experiments_cluster/` bandit directories (across
    SPY/BTC-USD/TLT/EURUSD=X/GC=F and all three policies) contain a PNG
    literally named "Score Softmax agent vs Average - Walk Forward Trai.png",
    including the UCB and Thompson directories.

    Also fixes the missing `plt.close(fig)` -- the original never closed a
    figure after saving (relying on `plt.show()` doing it implicitly under
    `%matplotlib inline`, which doesn't hold for a non-interactive/Agg backend
    such as CI or a batch script), so figures accumulated across a long run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
