"""Example: connecting backtester-mcp as an MCP server.

Configure your AI agent (Claude Desktop, etc.) with:

{
  "mcpServers": {
    "backtester-mcp": {
      "command": "backtester-mcp",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}

The agent can then call tools like:
- backtest_strategy(strategy_code="...", data_path="datasets/spy_daily.parquet")
- validate_robustness(strategy_code="...", data_path="datasets/spy_daily.parquet")
- optimize_parameters(strategy_code="...", data_path="...", param_space={...})
- compare_strategies([{code: "...", data_path: "..."}, ...])
"""

print("MCP server usage is via the CLI:")
print("  backtester-mcp serve --transport stdio")
print()
print("See the docstring in this file for Claude Desktop configuration.")
