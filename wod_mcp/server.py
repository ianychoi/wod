"""FastMCP server exposing the four WOD tools.

    Agent (LLM)  ->  MCP layer  ->  { ingest_data, analyze_exercise,
                                      generate_recs, deliver_feedback }
                                 ->  shared state store + per-run tracing

Every tool call is wrapped in trace_tool() so it is logged and correlated per
session. Run with:  python -m wod_mcp.server   (stdio transport)
"""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from . import trace
from .tools import analyze as analyze_mod
from .tools import deliver as deliver_mod
from .tools import ingest as ingest_mod
from .tools import recommend as recommend_mod

mcp = FastMCP("wod-exercise-feedback")


@mcp.tool()
def ingest_data(session_id: str, urls: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect workout records for a session into the shared state store.

    Pass `urls` to crawl live WOD posts; omit them to load the offline demo
    dataset (llmops/sample.json). Returns the source, count, and post ids.
    """
    with trace.trace_tool(session_id, "ingest_data", {"urls": bool(urls)}):
        return ingest_mod.ingest_data(session_id, urls)


@mcp.tool()
async def analyze_exercise(session_id: str) -> Dict[str, Any]:
    """Analyze the session's posts with OpenAI, Azure OpenAI, and Claude/Bedrock.

    Produces per-post is_exercise / exercise_summary / exercise_time /
    exercise_calories with a per-model breakdown, stored in shared state.
    """
    with trace.trace_tool(session_id, "analyze_exercise") as span:
        result = await analyze_mod.analyze_exercise(session_id)
        span["tokens"] = result.get("tokens")
        span["cost"] = result.get("cost")
        return result


@mcp.tool()
async def generate_recs(session_id: str) -> Dict[str, Any]:
    """Generate personalized Korean coaching feedback from the analysis results."""
    with trace.trace_tool(session_id, "generate_recs") as span:
        result = await recommend_mod.generate_recs(session_id)
        span["tokens"] = result.get("tokens")
        span["cost"] = result.get("cost")
        return result


@mcp.tool()
def deliver_feedback(session_id: str, dry_run: bool = True) -> Dict[str, Any]:
    """Publish feedback to the community. Dry-run by default; never posts live
    unless dry_run=False AND the WOD_DELIVER_ENABLED env flag is set."""
    with trace.trace_tool(session_id, "deliver_feedback", {"dry_run": dry_run}):
        return deliver_mod.deliver_feedback(session_id, dry_run)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
