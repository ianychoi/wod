"""deliver_feedback tool: publish personalized feedback to the community.

NEW capability (no prior code). This is the ONLY outward-facing action, so it
DEFAULTS TO DRY-RUN and never posts to a real community unless explicitly
enabled with both dry_run=False AND the WOD_DELIVER_ENABLED env flag. Real
posting is intentionally left a stub — wiring a live endpoint is out of scope
for the paper demo.
"""

import os
from typing import Any, Dict, List

from .. import state


def deliver_feedback(session_id: str, dry_run: bool = True) -> Dict[str, Any]:
    """Deliver state['recommendations']; record state['delivered'].

    dry_run=True (default): log what *would* be posted, mark dry_run=True.
    dry_run=False: only attempts real posting if WOD_DELIVER_ENABLED is set;
    otherwise it is refused and recorded as skipped. Real posting is a stub.

    Returns {"count", "dry_run", "live_enabled"}.
    """
    recommendations = state.get_state(session_id).get("recommendations", [])
    live_enabled = os.getenv("WOD_DELIVER_ENABLED", "").lower() in ("1", "true", "yes")

    delivered: List[Dict[str, Any]] = []
    for rec in recommendations:
        post_id = rec.get("post_id")
        if dry_run or not live_enabled:
            status = "dry_run" if dry_run else "skipped_no_live_flag"
            print(f"[deliver_feedback dry-run] post {post_id}: {rec.get('feedback_text')}")
            delivered.append({"post_id": post_id, "status": status, "dry_run": True})
        else:
            # STUB: a real implementation would POST to the community here.
            delivered.append({
                "post_id": post_id,
                "status": "not_implemented",
                "dry_run": False,
            })

    state.update_state(session_id, "delivered", delivered)
    return {
        "count": len(delivered),
        "dry_run": dry_run or not live_enabled,
        "live_enabled": live_enabled,
    }
