"""ingest_data tool: collect workout records into the run's shared state.

Wraps the Selenium crawl in lambda-container/app.py (crawl_posts). For a
reproducible offline demo it defaults to loading llmops/sample.json instead of
performing a live crawl, which needs WOD credentials + a Chrome/Selenium
container. Pass ``urls`` to force a live crawl.
"""

import json
from typing import Any, Dict, List, Optional

from .. import paths, state


def ingest_data(session_id: str, urls: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect posts for the session and store them under state['posts'].

    - ``urls`` given  -> live crawl via lambda-container/app.py:crawl_posts
    - ``urls`` empty  -> offline fallback: read llmops/sample.json

    Returns {"source", "count", "post_ids"}.
    """
    if urls:
        paths.ensure_importable()
        from app import crawl_posts  # lambda-container/app.py

        posts = crawl_posts(urls)
        source = "crawl"
    else:
        with open(paths.SAMPLE_JSON, "r", encoding="utf-8") as fh:
            posts = json.load(fh)
        source = "sample.json"

    state.update_state(session_id, "posts", posts)
    return {
        "source": source,
        "count": len(posts),
        "post_ids": [p.get("post_id") for p in posts],
    }
