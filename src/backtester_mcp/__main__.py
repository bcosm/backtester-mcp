"""Allow `python -m backtester_mcp` as a fallback when the entry-point script
isn't on PATH (common when the package is installed in a venv that the MCP
client, e.g. Claude Desktop, doesn't activate)."""

from backtester_mcp.cli import main

if __name__ == "__main__":
    main()
