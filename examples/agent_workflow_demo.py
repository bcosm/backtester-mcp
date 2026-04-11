"""Agent workflow documentation: the full MCP validation loop.

This file documents how an AI agent uses backtester-mcp to validate
a trading strategy. Each step shows the MCP tool call and expected
response structure.

Run this file to see the workflow printed:
    python examples/agent_workflow_demo.py
"""


def main():
    workflow = """
backtester-mcp Agent Workflow
=============================

An AI agent validates a trading strategy through these steps:

1. REGISTER DATASET
   Tool: register_dataset
   Input: {"file_path": "datasets/spy_daily.parquet", "name": "spy_daily"}
   Output: {"dataset_id": "abc-123", "name": "spy_daily",
            "rows": 2765, "columns": ["open","high","low","close","volume"]}

2. PROFILE DATASET
   Tool: profile_dataset
   Input: {"dataset_id": "abc-123"}
   Output: {"row_count": 2765, "frequency": "daily",
            "price_range": [102.5, 591.3], "price_type": "continuous",
            "ohlcv_available": {"open": true, "high": true, ...}}

3. GET STRATEGY TEMPLATE
   Tool: strategy_template
   Input: {"strategy_type": "momentum"}
   Output: {"code": "import numpy as np\\n..."}

4. BACKTEST
   Tool: backtest_strategy
   Input: {"strategy_code": "...", "data_path": "datasets/spy_daily.parquet",
           "fill_mode": "estimated"}
   Output: {"metrics": {"sharpe": 0.28, "total_return": 0.16, ...}}

5. VALIDATE (full pipeline)
   Tool: validate_strategy
   Input: {"strategy_code": "...", "data_path": "...",
           "params": {"fast_period": 10, "slow_period": 50}}
   Output: {"verdict": "caution",
            "reasons": ["bootstrap CI includes zero"],
            "metrics": {...},
            "bootstrap_sharpe": {"sharpe": 0.28, "ci_lower": -0.31, ...},
            "pbo": {"pbo": 0.36, ...},
            "scenarios": {"optimistic": {...}, "base": {...}, ...}}

6. OPTIMIZE
   Tool: optimize_parameters
   Input: {"strategy_code": "...", "data_path": "...",
           "param_space": {"fast_period": [5, 30], "slow_period": [20, 100]}}
   Output: {"best_params": {"fast_period": 12, "slow_period": 45},
            "best_metric": 0.42, "pbo": 0.48}

7. SAVE RUN
   Tool: save_run
   Input: {"strategy_name": "momentum_v1", "metrics": {...}}
   Output: {"run_id": "def-456"}

8. COMPARE RUNS
   Tool: compare_runs
   Input: {"run_ids": ["def-456", "ghi-789"]}
   Output: {"runs": [{"strategy_name": "momentum_v1", "metrics": {...}},
                      {"strategy_name": "momentum_v2", "metrics": {...}}]}

9. GENERATE REPORT
   Tool: generate_report
   Input: {"run_id": "def-456", "output_path": "report.html"}
   Output: {"report_path": "report.html"}

The agent uses the verdict from step 5 to decide whether to trust
the strategy. A "pass" verdict means all checks cleared. A "caution"
verdict lists specific reasons for concern.
"""
    print(workflow)


if __name__ == "__main__":
    main()
