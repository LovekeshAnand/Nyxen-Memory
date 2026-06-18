"""Load and parse official LoCoMo locomo10.json dataset."""
import json
import re
from typing import Any, Dict, List, Optional


def _parse_session_idx(dia_id: str) -> Optional[int]:
    """Parse 'D3:12' -> session 3."""
    if not dia_id or ":" not in dia_id:
        return None
    session_part = dia_id.split(":")[0]
    if session_part.startswith("D"):
        try:
            return int(session_part[1:])
        except ValueError:
            return None
    return None


def format_session_text(session_turns: List[Dict[str, Any]], session_idx: int, date_time: str = "") -> str:
    lines = [f"--- Session {session_idx} ({date_time}) ---"]
    for turn in session_turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", "")
        if text:
            lines.append(f"[{dia_id}] {speaker}: {text}")
    return "\n".join(lines)


def load_locomo_dataset(path: str) -> List[Dict[str, Any]]:
    """Load locomo10.json and return list of conversation records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    raise ValueError(f"Unexpected LoCoMo JSON structure in {path}")


def parse_conversation(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse one LoCoMo sample into a canonical internal format.

    Returns dict with: sample_id, speaker_a, speaker_b, turns, sessions, qa, observations
    """
    conv = record.get("conversation", record)
    sample_id = record.get("sample_id") or conv.get("sample_id") or "unknown"

    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")

    sessions = []
    turns = []
    turn_id = 0

    session_keys = sorted(
        [k for k in conv.keys() if re.match(r"session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    for session_key in session_keys:
        session_idx = int(session_key.split("_")[1])
        date_key = f"session_{session_idx}_date_time"
        date_time = conv.get(date_key, "")
        session_turns = conv.get(session_key, [])

        formatted = format_session_text(session_turns, session_idx, date_time)
        sessions.append({
            "session_idx": session_idx,
            "date_time": date_time,
            "formatted_text": formatted,
            "turns": session_turns,
        })

        for turn in session_turns:
            turn_id += 1
            text = turn.get("text", "")
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id", "")
            turns.append({
                "turn_id": turn_id,
                "session_idx": session_idx,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": text,
                "user_text": text if speaker == speaker_a else "",
                "assistant_text": text if speaker != speaker_a else "",
                "formatted_line": f"[{dia_id}] {speaker}: {text}",
            })

    qa_list = record.get("qa", [])
    observations = record.get("observation", []) or conv.get("observation", [])

    return {
        "sample_id": sample_id,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "sessions": sessions,
        "turns": turns,
        "qa": qa_list,
        "observations": observations,
        "evidence_session_map": _build_evidence_map(qa_list),
    }


def _build_evidence_map(qa_list: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Map question index to evidence session indices."""
    result = {}
    for idx, qa in enumerate(qa_list):
        sessions = set()
        for ev in qa.get("evidence", []):
            sidx = _parse_session_idx(ev)
            if sidx is not None:
                sessions.add(sidx)
        result[str(idx)] = sorted(sessions)
    return result


def filter_qa(
    qa_list: List[Dict[str, Any]],
    categories: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Filter QA items by category list (default: scored categories 1-4)."""
    if categories is None:
        categories = [1, 2, 3, 4]
    return [qa for qa in qa_list if int(qa.get("category", 0)) in categories]


def load_all_conversations(path: str) -> List[Dict[str, Any]]:
    """Load and parse all conversations from locomo10.json."""
    raw = load_locomo_dataset(path)
    return [parse_conversation(r) for r in raw]
