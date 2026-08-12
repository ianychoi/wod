"""analyze_exercise tool: multi-LLM exercise analysis over the run's posts.

Wraps AIModelComparator from llmops/main.py, reusing its chat clients,
encode_image / prepare_messages / analyze_with_model / compare_models, and the
Korean analysis prompt that requests JSON fields is_exercise / exercise_summary
/ exercise_time (min) / exercise_calories (kcal).

Output is stored per-post with a per_model breakdown so it maps 1:1 onto the
evaluation CSV columns. analysis_to_csv_rows() emits that exact schema so
evaluation/evaluate.py keeps working unchanged.
"""

import json
import os
from typing import Any, Dict, List

from .. import paths, state

# Maps our internal model keys to AIModelComparator's model display names and to
# the evaluation CSV column prefixes.
_MODEL_KEYS = {
    "openai": "OpenAI",
    "aoai": "Azure OpenAI",
    "bedrock": "Claude (Bedrock)",
    "llama": "Llama (Bedrock)",
}

# Header of evaluation/exercise_analysis-from-741-posts.csv (kept verbatim), with
# the llama_* triple appended for the added Bedrock Llama model.
CSV_HEADER = [
    "post_id",
    "human_is_exercise", "human_time", "human_calories",
    "duration_estimation", "calorie_estimation",
    "openai_is_exercise", "openai_exercise_time", "openai_exercise_calories",
    "aoai_is_exercise", "aoai_exercise_time", "aoai_exercise_calories",
    "bedrock_is_exercise", "bedrock_exercise_time", "bedrock_exercise_calories",
    "llama_is_exercise", "llama_exercise_time", "llama_exercise_calories",
]


def _build_prompt(post: Dict[str, Any]) -> str:
    """Recreate the analysis prompt from llmops/main.py:process_entry."""
    return f"""
    포스트 ID: {post.get('post_id')}
    포스트 일자: {post.get('post_date')}
    사용자 ID: {post.get('actor_id')}
    사용자 이름: {post.get('name')}
    오늘의 운동 기록:
    - {post.get('content')}
    첨부 이미지 파일과 함께 이 운동 포스트 내역을 분석해주세요.
    결과는 포스트 ID 값, 포스트 일자 값, 사용자 ID 값, 사용자 이름을 포함해서 JSON 형태로 출력해줘.
    또한 해당 포스트 내용이 운동과 관련있는지 여부를 체크해서 is_exercise 필드에 True 또는 False로 표시해줘.
    운동에 대한 요약을 exercise_summary 필드에 담아서 출력해줘.
    운동 요약을 분석해서 운동 시간을 분 (minute) 단위로 예상후 exercise_time 필드에 숫자로 출력해줘.
    그리고 운동 요약을 분석해서 소모한 칼로리를 kcal 단위로 예상해서 exercise_calories 필드에 숫자로 출력해줘.
    그 외 모든 데이터를 JSON 필드에 담아 결과는 완전한 JSON 형태, 마크다운을 사용하지 않고 JSON Text만 출력해줘.
    """


def _resolve_photos(post: Dict[str, Any]) -> List[str]:
    """Resolve photo filenames to absolute paths under llmops/sample_photos/."""
    resolved = []
    for photo in post.get("photos", []):
        # Already a path? keep it; else look under sample_photos/.
        candidate = photo if os.path.isabs(photo) else os.path.join(
            paths.SAMPLE_PHOTOS_DIR, os.path.basename(photo)
        )
        if os.path.exists(candidate):
            resolved.append(candidate)
    return resolved


async def analyze_exercise(session_id: str) -> Dict[str, Any]:
    """Analyze each post in state['posts'] with all three models.

    Writes state['analysis'] as a list of::

        {post_id, is_exercise, exercise_summary, exercise_time,
         exercise_calories, per_model: {openai|aoai|bedrock: {...}}}

    The top-level is_exercise/time/calories use the Bedrock (Claude) result as
    the primary answer, falling back to whichever model returned first.

    Returns {"count", "analyzed_post_ids", "tokens", "cost"} — tokens/cost are
    summed across all model calls so the tracing layer can record LLMOps
    metrics.
    """
    paths.ensure_importable()
    from main import AIModelComparator  # llmops/main.py

    comparator = AIModelComparator()
    posts = state.get_state(session_id).get("posts", [])

    analysis: List[Dict[str, Any]] = []
    total_tokens = 0
    total_cost = 0.0

    for post in posts:
        prompt = _build_prompt(post)
        image_paths = _resolve_photos(post)
        results = await comparator.compare_models(prompt, image_paths)

        per_model: Dict[str, Any] = {}
        for key, display_name in _MODEL_KEYS.items():
            result = results.get(display_name, {})
            parsed = _parse_result(result)
            per_model[key] = parsed
            total_tokens += parsed.get("total_tokens") or 0
            total_cost += parsed.get("total_cost") or 0.0

        primary = per_model.get("bedrock") or next(iter(per_model.values()), {})
        analysis.append({
            "post_id": post.get("post_id"),
            "is_exercise": primary.get("is_exercise"),
            "exercise_summary": primary.get("exercise_summary"),
            "exercise_time": primary.get("exercise_time"),
            "exercise_calories": primary.get("exercise_calories"),
            "per_model": per_model,
        })

    state.update_state(session_id, "analysis", analysis)
    return {
        "count": len(analysis),
        "analyzed_post_ids": [a["post_id"] for a in analysis],
        "tokens": total_tokens,
        "cost": total_cost,
    }


def _response_text(response: Any) -> str:
    """Flatten a chat response into plain text.

    Converse-style models (Claude Opus, Llama on Bedrock) may return content as a
    list of blocks (e.g. [{"type": "text", "text": "..."}]) rather than a string,
    so join the text blocks. Also strips ```json code fences if present.
    """
    if isinstance(response, list):
        parts = []
        for block in response:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        text = "".join(parts)
    else:
        text = response or ""
    text = text.strip()
    if text.startswith("```"):
        # drop the opening fence line (```json / ```) and the trailing fence
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _parse_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one AIModelComparator model result into a flat per-model record."""
    out: Dict[str, Any] = {
        "total_tokens": result.get("total_tokens"),
        "total_cost": result.get("total_cost"),
        "time_taken": result.get("time_taken"),
    }
    if "error" in result:
        out["error"] = result["error"]
        return out
    try:
        parsed = json.loads(_response_text(result.get("response", "")))
        out["is_exercise"] = parsed.get("is_exercise")
        out["exercise_summary"] = parsed.get("exercise_summary")
        out["exercise_time"] = parsed.get("exercise_time")
        out["exercise_calories"] = parsed.get("exercise_calories")
    except (json.JSONDecodeError, TypeError) as exc:
        out["error"] = f"JSON parse error: {exc}"
        out["raw_response"] = result.get("response")
    return out


def analysis_to_csv_rows(session_id: str) -> List[Dict[str, Any]]:
    """Convert state['analysis'] into rows matching CSV_HEADER for evaluate.py.

    Human-label and estimation-flag columns are left blank (they come from human
    annotation, not the model); the three model triples are filled from
    per_model. This keeps evaluation/evaluate.py runnable against model output.
    """
    rows = []
    for item in state.get_state(session_id).get("analysis", []):
        row = {col: "" for col in CSV_HEADER}
        row["post_id"] = item.get("post_id")
        for key in _MODEL_KEYS:
            pm = item.get("per_model", {}).get(key, {})
            row[f"{key}_is_exercise"] = pm.get("is_exercise")
            row[f"{key}_exercise_time"] = pm.get("exercise_time")
            row[f"{key}_exercise_calories"] = pm.get("exercise_calories")
        rows.append(row)
    return rows
