"""Session-keyed shared state store.

Every MCP tool reads and writes the run context for its ``session_id``. The
diagram calls this the "Shared state store: run context keyed by session".

For a paper/demo this is deliberately simple: an in-process dict cache backed by
one JSON file per session under ``wod_mcp/runs/<session_id>.json`` so a completed
run is fully inspectable afterwards. Not a database — that is fine at this scale.

Run-context shape::

    {
      "session_id": str,
      "posts":           [ {post_id, actor_id, name, content, post_date, photos:[...]} ],
      "analysis":        [ {post_id, is_exercise, exercise_summary,
                            exercise_time, exercise_calories, per_model:{...}} ],
      "recommendations": [ {post_id, feedback_text} ],
      "delivered":       [ {post_id, status, dry_run} ]
    }

The ``posts`` shape mirrors llmops/sample.json; ``analysis[].per_model`` maps 1:1
onto the evaluation CSV columns (see wod_mcp/tools/analyze.py).
"""

import json
import os
from typing import Any, Dict

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

# Top-level keys a run context may hold, seeded empty on first touch.
_STAGE_KEYS = ("posts", "analysis", "recommendations", "delivered")

# In-process cache: session_id -> run context dict.
_CACHE: Dict[str, Dict[str, Any]] = {}


def _path(session_id: str) -> str:
    return os.path.join(RUNS_DIR, f"{session_id}.json")


def _empty(session_id: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"session_id": session_id}
    for key in _STAGE_KEYS:
        ctx[key] = []
    return ctx


def get_state(session_id: str) -> Dict[str, Any]:
    """Return the run context for ``session_id``, loading from disk if present.

    Never returns None: an unknown session is seeded with an empty context.
    """
    if session_id in _CACHE:
        return _CACHE[session_id]

    path = _path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            ctx = json.load(fh)
    else:
        ctx = _empty(session_id)

    _CACHE[session_id] = ctx
    return ctx


def update_state(session_id: str, key: str, value: Any) -> Dict[str, Any]:
    """Set ``ctx[key] = value`` for the session and persist. Returns the context."""
    ctx = get_state(session_id)
    ctx[key] = value
    save(session_id)
    return ctx


def save(session_id: str) -> str:
    """Persist the session's run context to disk and return the file path."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = _path(session_id)
    ctx = get_state(session_id)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ctx, fh, ensure_ascii=False, indent=2)
    return path
