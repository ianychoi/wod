"""wod_mcp: MCP-based agent architecture for the WOD exercise-feedback system.

Wraps the existing crawl (lambda-container) and multi-LLM analysis (llmops)
behind four MCP tools orchestrated by an LLM agent, with a session-keyed shared
state store and per-run tracing. Backs arXiv:2501.13723.

The package is named ``wod_mcp`` (not ``mcp``) so it does not shadow the
installed Model Context Protocol SDK package.
"""
