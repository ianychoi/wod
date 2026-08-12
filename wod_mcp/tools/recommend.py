"""generate_recs tool: personalized Korean feedback from analysis results.

NEW capability (no prior code). Reuses AIModelComparator's chat client to turn
each post's exercise analysis into a short, encouraging personalized coaching
message. Writes state['recommendations'].
"""

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from .. import paths, state

_COACH_SYSTEM = SystemMessage(
    content="당신은 사용자의 운동 기록을 보고 짧고 친근한 맞춤형 피드백을 제공하는 전문 트레이너입니다."
)


def _coach_prompt(item: Dict[str, Any]) -> str:
    return f"""
    아래 운동 분석 결과를 바탕으로 사용자에게 2~3문장의 격려와 맞춤형 조언을 한국어로 작성해줘.
    - 운동 여부: {item.get('is_exercise')}
    - 운동 요약: {item.get('exercise_summary')}
    - 예상 운동 시간(분): {item.get('exercise_time')}
    - 예상 소모 칼로리(kcal): {item.get('exercise_calories')}
    마크다운 없이 순수 텍스트로만 출력해줘.
    """


async def generate_recs(session_id: str) -> Dict[str, Any]:
    """Produce a feedback_text per analyzed post; store state['recommendations'].

    Returns {"count", "post_ids", "tokens", "cost"}.
    """
    paths.ensure_importable()
    from main import AIModelComparator  # llmops/main.py

    comparator = AIModelComparator()
    # Reuse the same chat client used elsewhere; Bedrock/Claude is the primary.
    model = comparator.bedrock_chat

    analysis = state.get_state(session_id).get("analysis", [])
    recommendations: List[Dict[str, Any]] = []
    total_tokens = 0
    total_cost = 0.0

    for item in analysis:
        messages = [_COACH_SYSTEM, HumanMessage(content=_coach_prompt(item))]
        try:
            response = await model.ainvoke(messages)
            feedback_text = response.content
            usage = getattr(response, "usage_metadata", None) or {}
            total_tokens += usage.get("total_tokens", 0) or 0
        except Exception as exc:  # noqa: BLE001 - degrade gracefully in demo
            feedback_text = f"[feedback unavailable: {exc}]"
        recommendations.append({
            "post_id": item.get("post_id"),
            "feedback_text": feedback_text,
        })

    state.update_state(session_id, "recommendations", recommendations)
    return {
        "count": len(recommendations),
        "post_ids": [r["post_id"] for r in recommendations],
        "tokens": total_tokens,
        "cost": total_cost,
    }
