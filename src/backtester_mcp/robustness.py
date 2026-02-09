"""Overfitting detection: PBO, bootstrap Sharpe CI, deflated Sharpe ratio."""

import itertools
import numpy as np
from backtester_mcp.metrics import sharpe, TRADING_DAYS


def pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    returns_matrix: (n_periods, n_strategies) array of returns.
    Each column is a different strategy or parameter set.
    """
    n_periods, n_strats = returns_matrix.shape
    if n_strats < 2:
        raise ValueError("need at least 2 strategies to compute PBO")

    block_size = n_periods // n_splits
    if block_size < 2:
        raise ValueError("not enough data for requested n_splits")

    # split into blocks
    blocks = []
    for i in range(n_splits):
        start = i * block_size
        end = start + block_size
        blocks.append(returns_matrix[start:end])

    half = n_splits // 2
    combos = list(itertools.combinations(range(n_splits), half))

    # cap combinations to keep runtime sane
    rng = np.random.default_rng(42)
    if len(combos) > 500:
        idx = rng.choice(len(combos), 500, replace=False)
        combos = [combos[i] for i in idx]

    all_indices = set(range(n_splits))
    underperform_count = 0

    logits = []

    for is_indices in combos:
        oos_indices = sorted(all_indices - set(is_indices))

        is_returns = np.vstack([blocks[i] for i in is_indices])
        oos_returns = np.vstack([blocks[i] for i in oos_indices])

        # rank strategies by in-sample Sharpe
        is_sharpes = np.array([sharpe(is_returns[:, j]) for j in range(n_strats)])
        best_is = np.argmax(is_sharpes)

        # check OOS performance of the IS-best strategy
        oos_sharpes = np.array([sharpe(oos_returns[:, j]) for j in range(n_strats)])
        oos_rank = np.sum(oos_sharpes >= oos_sharpes[best_is])

        # relative rank (1 = best, n_strats = worst)
        w = oos_rank / n_strats

        if w > 0.5:
            underperform_count += 1

        # logit for distribution — clamp to avoid log(0)
        w_clamped = np.clip(w, 0.01, 0.99)
        logits.append(np.log(w_clamped / (1 - w_clamped)))

    pbo_score = underperform_count / len(combos)
    return {
        "pbo": round(pbo_score, 4),
        "n_combinations": len(combos),
        "logits": np.array(logits),
    }


def bootstrap_sharpe(returns: np.ndarray, n_samples: int = 10_000,
                     ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap confidence interval for the Sharpe ratio."""
    rng = np.random.default_rng(seed)
    n = len(returns)

    sharpes = np.empty(n_samples)
    for i in range(n_samples):
        sample = rng.choice(returns, size=n, replace=True)
        sharpes[i] = sharpe(sample)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(sharpes, alpha * 100))
    hi = float(np.percentile(sharpes, (1 - alpha) * 100))
    point = sharpe(returns)

    return {
        "sharpe": round(point, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "ci_includes_zero": lo <= 0 <= hi,
        "distribution": sharpes,
    }


def deflated_sharpe(observed_sharpe: float, n_returns: int,
                    n_strategies: int, skew: float = 0.0,
                    kurtosis: float = 3.0) -> dict:
    """Deflated Sharpe Ratio — accounts for multiple testing.

    From Lopez de Prado (2014). Tests whether the observed Sharpe is
    significantly above what you'd expect from the best of n_strategies
    independent trials on white noise.
    """
    from scipy import stats

    # expected max Sharpe under null (Euler-Mascheroni approximation)
    gamma = 0.5772156649
    e_max = ((1 - gamma) * stats.norm.ppf(1 - 1 / n_strategies)
             + gamma * stats.norm.ppf(1 - 1 / (n_strategies * np.e)))

    # standard error of Sharpe estimator with higher moments
    se = np.sqrt(
        (1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
        / (n_returns - 1)
    )

    if se == 0:
        return {"dsr": 0.0, "p_value": 1.0}

    test_stat = (observed_sharpe - e_max) / se
    p_value = stats.norm.cdf(test_stat)

    return {
        "dsr": round(float(test_stat), 4),
        "p_value": round(float(1 - p_value), 4),
        "expected_max_sharpe": round(float(e_max), 4),
    }
