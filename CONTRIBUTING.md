# Contributing

## Setup

```bash
git clone https://github.com/bcosm/backtester-mcp.git
cd backtester-mcp
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- snake_case everywhere
- Comments explain *why*, not *what*
- Type hints on public functions
- Lines <= 99 characters
- No `TODO`, `FIXME`, or commented-out code

## Adding a Strategy

Create a file in `strategies/` with:

```python
import numpy as np

DEFAULT_PARAMS = {"lookback": 20}

def generate_signals(prices, lookback=20):
    signals = np.zeros(len(prices))
    # your logic here
    return signals
```

## Adding a Robustness Method

Add the function to `src/backtester_mcp/robustness.py`. It should:
- Accept a returns array or strategy function
- Return a dict with clear, named fields
- Be grounded in published research (cite the paper)
- Have at least one test in `tests/test_robustness.py`

## Adding an MCP Tool

Add the tool in `src/backtester_mcp/mcp_server.py` using `@mcp.tool()`. Include:
- Clear docstring (this becomes the tool description for agents)
- Structured JSON response via `_make_response()`
- `warnings` for any issues detected
- A test in `tests/test_mcp_integration.py`

## Commits

- Atomic: one logical change per commit
- Imperative mood: "add feature" not "added feature"
- Each commit should leave tests passing
