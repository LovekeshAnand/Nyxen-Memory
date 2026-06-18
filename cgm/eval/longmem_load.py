"""Load official LongMemEval JSON datasets."""
import json
from typing import Any, Dict, List


def load_longmemeval_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected LongMemEval JSON structure in {path}")


def parse_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one LongMemEval instance."""
    return {
        "question_id": instance["question_id"],
        "question_type": instance.get("question_type", "unknown"),
        "question": instance["question"],
        "answer": instance.get("answer", ""),
        "question_date": instance.get("question_date", ""),
        "haystack_sessions": instance.get("haystack_sessions", []),
        "haystack_dates": instance.get("haystack_dates", []),
        "haystack_session_ids": instance.get("haystack_session_ids", []),
        "answer_session_ids": instance.get("answer_session_ids", []),
        "is_abstention": instance.get("question_id", "").endswith("_abs")
            or instance.get("question_type", "").endswith("_abs"),
    }


def load_all_instances(path: str) -> List[Dict[str, Any]]:
    return [parse_instance(i) for i in load_longmemeval_dataset(path)]
