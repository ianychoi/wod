"""Agent (LLM) loop that orchestrates the four MCP tools.

Implements the top box of the target architecture: an LLM that plans, selects
tools, and re-plans on failure. It launches wod_mcp.server over stdio, discovers
the four tools via langchain-mcp-adapters, and drives them in order
(ingest_data -> analyze_exercise -> generate_recs -> deliver_feedback) for one
session, retrying/skipping on tool failure.

Usage:
    python -m wod_mcp.agent --session demo-001
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from . import paths, state, trace


def _build_model():
    """Reuse the Bedrock/Claude chat client configured for the project."""
    paths.ensure_importable()
    from main import AIModelComparator  # llmops/main.py
    return AIModelComparator().bedrock_chat


async def run_agent(session_id: str) -> dict:
    """Discover the MCP tools and let the agent orchestrate the run.

    Returns the final shared-state context for the session.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain.agents import create_agent

    load_dotenv()

    server_cfg = {
        "wod": {
            "command": sys.executable,
            "args": ["-m", "wod_mcp.server"],
            "transport": "stdio",
            # Run the server subprocess from the repo root so its imports work.
            "cwd": paths.REPO_ROOT,
            "env": dict(os.environ),
        }
    }

    client = MultiServerMCPClient(server_cfg)
    tools = await client.get_tools()

    system_prompt = (
        "You orchestrate a workout-analysis pipeline for one session. "
        f"The session_id is '{session_id}' — pass it to every tool call. "
        "Run the tools in this order: ingest_data, analyze_exercise, "
        "generate_recs, then deliver_feedback (keep dry_run=true). "
        "If a tool fails, retry once, then skip it and continue. "
        "When all steps are done, briefly summarize what happened."
    )

    agent = create_agent(_build_model(), tools, system_prompt=system_prompt)
    await agent.ainvoke({
        "messages": [
            {"role": "user", "content": f"Process session {session_id} end to end."},
        ]
    })

    return state.get_state(session_id)


def main():
    parser = argparse.ArgumentParser(description="Run the WOD MCP agent for a session.")
    parser.add_argument("--session", default="demo-001", help="session id")
    args = parser.parse_args()

    ctx = asyncio.run(run_agent(args.session))

    print(f"\n=== Run complete for session {args.session} ===")
    print(f"posts:           {len(ctx.get('posts', []))}")
    print(f"analysis:        {len(ctx.get('analysis', []))}")
    print(f"recommendations: {len(ctx.get('recommendations', []))}")
    print(f"delivered:       {len(ctx.get('delivered', []))}")
    print(f"state file:      {state.save(args.session)}")
    print(f"trace records:   {len(trace.read_trace(args.session))}")


if __name__ == "__main__":
    main()
