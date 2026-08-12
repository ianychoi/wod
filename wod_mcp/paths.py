"""Repo path helpers.

The reusable logic we wrap lives in sibling top-level dirs (llmops/,
lambda-container/) that are not importable packages. This module puts them on
sys.path so wod_mcp can import them, and exposes well-known file locations.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLMOPS_DIR = os.path.join(REPO_ROOT, "llmops")
LAMBDA_DIR = os.path.join(REPO_ROOT, "lambda-container")

SAMPLE_JSON = os.path.join(LLMOPS_DIR, "sample.json")
SAMPLE_PHOTOS_DIR = os.path.join(LLMOPS_DIR, "sample_photos")


def ensure_importable():
    """Add llmops/ and lambda-container/ to sys.path (idempotent)."""
    for path in (LLMOPS_DIR, LAMBDA_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
