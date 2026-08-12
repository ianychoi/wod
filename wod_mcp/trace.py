"""Per-run tracing / audit.

The diagram calls this "Tracing and audit: every tool call logged, correlated
per run". Each tool call appends one JSON line to
``wod_mcp/runs/<session_id>.trace.jsonl``::

    {seq, session_id, tool, args_summary, started_at, duration_s,
     ok, error?, tokens?, cost?}

``tokens``/``cost`` reuse the LLMOps metrics already produced by
``AIModelComparator.analyze_with_model`` (total_tokens, total_cost, time_taken),
so analyze traces carry real cost data.

Clock note: the caller passes ``started_at``/``duration_s`` (or lets the context
manager stamp them with time.time()). ``trace_tool`` accepts an optional ``now``
callable so a deterministic replay can inject timestamps instead of the wall
clock.
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

# Monotonic per-session sequence counters for correlating calls within a run.
_SEQ: Dict[str, int] = {}


def _trace_path(session_id: str) -> str:
    return os.path.join(RUNS_DIR, f"{session_id}.trace.jsonl")


def _next_seq(session_id: str) -> int:
    seq = _SEQ.get(session_id, 0) + 1
    _SEQ[session_id] = seq
    return seq


def _append(session_id: str, record: Dict[str, Any]) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(_trace_path(session_id), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def trace_tool(
    session_id: str,
    tool: str,
    args_summary: Optional[Dict[str, Any]] = None,
    now: Callable[[], float] = time.time,
):
    """Context manager wrapping one tool call; writes one trace record on exit.

    Attach LLMOps metrics from inside the ``with`` block via the yielded dict::

        with trace_tool(sid, "analyze_exercise", {...}) as span:
            ...
            span["tokens"] = total_tokens
            span["cost"] = total_cost

    Records ok=False plus the error string if the block raises, then re-raises.
    """
    span: Dict[str, Any] = {}
    started_at = now()
    seq = _next_seq(session_id)
    record: Dict[str, Any] = {
        "seq": seq,
        "session_id": session_id,
        "tool": tool,
        "args_summary": args_summary or {},
        "started_at": started_at,
    }
    try:
        yield span
    except Exception as exc:  # noqa: BLE001 - record then re-raise
        record["ok"] = False
        record["error"] = str(exc)
        raise
    else:
        record["ok"] = True
    finally:
        record["duration_s"] = now() - started_at
        # Optional LLMOps metrics set by the caller on the span.
        for metric in ("tokens", "cost"):
            if metric in span:
                record[metric] = span[metric]
        _append(session_id, record)


def read_trace(session_id: str):
    """Return the list of trace records for a session (empty if none)."""
    path = _trace_path(session_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
