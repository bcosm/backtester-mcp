"""Parameter optimization with Optuna + automatic PBO check."""

import numpy as np
import optuna
from backtester_mcp.engine import backtest
from backtester_mcp.robustness import pbo


# suppress optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(
    strategy_fn,
    prices: np.ndarray,
    param_space: dict[str, tuple[int | float, int | float]],
    metric: str = "sharpe",
    n_trials: int = 100,
    check_pbo: bool = True,
    pbo_splits: int = 16,
    seed: int = 42,
) -> dict:
    """Optimize strategy parameters via Bayesian search.

    strategy_fn: callable(prices, **params) -> signals array
    param_space: {"param_name": (low, high)} — int bounds use int sampling
    """
    # collect returns from each trial for PBO
    trial_returns = []
    trial_params = []

    def objective(trial):
        params = {}
        for name, (lo, hi) in param_space.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, float(lo), float(hi))

        signals = strategy_fn(prices, **params)
        result = backtest(prices, signals)
        trial_returns.append(result.returns.copy())
        trial_params.append(params)
        return result.metrics[metric]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_idx = study.best_trial.number
    best_params = trial_params[best_idx]

    out = {
        "best_params": best_params,
        "best_metric": round(study.best_value, 4),
        "metric_name": metric,
        "n_trials": n_trials,
    }

    if check_pbo and len(trial_returns) >= 2:
        # trim all return arrays to same length
        min_len = min(len(r) for r in trial_returns)
        matrix = np.column_stack([r[:min_len] for r in trial_returns])
        pbo_result = pbo(matrix, n_splits=min(pbo_splits, min_len // 2))
        out["pbo"] = pbo_result["pbo"]
        out["pbo_n_combinations"] = pbo_result["n_combinations"]

    return out
