# wod_mcp — MCP-based agent architecture

An MCP refactor of the WOD exercise-feedback pipeline for the paper
*Intelligent Exercise and Feedback System for Social Healthcare using LLMOps*
(arXiv:2501.13723). It turns the old linear script pipeline into an **LLM agent
orchestrating four MCP tools** over a shared, per-session state store with full
tracing.

```
Agent (LLM)  ─►  MCP layer  ─►  ingest_data ─► analyze_exercise ─► generate_recs ─► deliver_feedback
   plans /                      │
   re-plans                     ├─ Shared state store   wod_mcp/runs/<session>.json
   on failure                   └─ Tracing / audit      wod_mcp/runs/<session>.trace.jsonl
```

The tools are thin wrappers over existing code — `lambda-container/app.py`
(`crawl_posts`) for ingest and `llmops/main.py` (`AIModelComparator`) for
analysis — so results stay compatible with `evaluation/evaluate.py`.

> The package is named `wod_mcp` (not `mcp`) so it does not shadow the installed
> Model Context Protocol SDK.

## Setup

```sh
pip install -r wod_mcp/requirements.txt
cp wod_mcp/.env-example wod_mcp/.env   # fill in LLM credentials
```

## Run

Run the server on its own (stdio):

```sh
python -m wod_mcp.server
```

Or run the agent, which launches the server and orchestrates a session end to
end (offline: uses `llmops/sample.json` as the ingest source):

```sh
python -m wod_mcp.agent --session demo-001
```

After a run, inspect:

- `wod_mcp/runs/demo-001.json` — shared state (posts → analysis →
  recommendations → delivered)
- `wod_mcp/runs/demo-001.trace.jsonl` — one correlated record per tool call,
  with token/cost metrics on the LLM tools

## The four tools

| Tool | Wraps | Reads → Writes state |
|------|-------|----------------------|
| `ingest_data(session_id, urls?)` | `crawl_posts` / `sample.json` | — → `posts` |
| `analyze_exercise(session_id)` | `AIModelComparator` | `posts` → `analysis` |
| `generate_recs(session_id)` | Bedrock/Claude chat | `analysis` → `recommendations` |
| `deliver_feedback(session_id, dry_run=True)` | stub publisher | `recommendations` → `delivered` |

`deliver_feedback` is **dry-run by default** and never posts to a real community
unless `dry_run=False` *and* `WOD_DELIVER_ENABLED` is set. Real posting is a
deliberate stub.

## Evaluation compatibility

`wod_mcp/tools/analyze.py:analysis_to_csv_rows(session_id)` emits rows matching
the exact header of `evaluation/exercise_analysis-from-741-posts.csv`, so model
output can be scored with the unchanged `evaluation/evaluate.py`.
