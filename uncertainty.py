"""Binomial uncertainty and exploratory cell-versus-rest comparisons."""

import numpy as np
from scipy.stats import fisher_exact, norm


def wilson_interval(errors, total, confidence=.95):
    """Two-sided Wilson score interval; an empty cell has no estimate or interval."""
    errors, total = np.asarray(errors, dtype=float), np.asarray(total, dtype=float)
    if not 0 < confidence < 1 or np.any(total < 0) or np.any(errors < 0) or np.any(errors > total):
        raise ValueError("Use 0 <= errors <= total and a confidence level between 0 and 1.")
    n = np.where(total > 0, total, np.nan)
    z = norm.ppf((1 + confidence) / 2)
    p = errors / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return np.clip(center-half, 0, 1), np.clip(center+half, 0, 1)


def add_cell_uncertainty(cells):
    cells = cells.copy()
    cells["error_ci_low"], cells["error_ci_high"] = wilson_interval(cells.errors, cells.total)
    n, e = int(cells.total.sum()), int(cells.errors.sum())
    pvalues = []
    for row in cells.itertuples():
        rest_n, rest_e = n - row.total, e - row.errors
        pvalues.append(float(fisher_exact([[row.errors, row.total-row.errors], [rest_e, rest_n-rest_e]], alternative="greater").pvalue)
                       if row.total > 0 and rest_n > 0 else np.nan)
    cells["hotspot_p"] = pvalues
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(np.asarray(pvalues)[valid], kind="stable")]
    adjusted = np.full(len(cells), np.nan)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(np.asarray(pvalues)[order] * np.arange(len(order), 0, -1)))
    cells["hotspot_p_adjusted"] = adjusted
    return cells


def evidence_labels(cells, small_n=10):
    return np.select(
        [cells.total == 0, cells.total < small_n, cells.hotspot_p_adjusted < .05],
        ["No examples", "Too few examples", "Elevated error evidence"],
        default="No clear elevation",
    )
