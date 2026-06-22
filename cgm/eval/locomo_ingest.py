"""Ingest LoCoMo conversations into the CGM SQLite store and RAG index."""
import re
from typing import Any, Dict, List, Optional

from cgm.database.schema import HybridMemoryObject


def _normalize_relative_date(content: str, session_date: str) -> Optional[str]:
    """Convert simple relative dates into an explicit date anchor when possible."""
    content_lower = content.lower()
    if not session_date:
        return None
    try:
        from datetime import datetime, timedelta

        date_match = re.search(r"(\d{1,2}\s+[A-Za-z]+\s*,?\s*\d{4})", session_date)
        if not date_match:
            return None
        date_text = date_match.group(1).replace(",", "")
        base = datetime.strptime(date_text, "%d %B %Y")
        if "yesterday" in content_lower:
            result = base - timedelta(days=1)
            return f"{result.day} {result.strftime('%B %Y')}"
        if "today" in content_lower:
            return f"{base.day} {base.strftime('%B %Y')}"
    except Exception:
        return None
    return None


def _simple_extract_triples(text: str, speaker_a: str, session_date: str = "") -> List[List[str]]:
    """
    Enhanced triple extraction for LoCoMo benchmark ingest.
    Handles facts, events, preferences, dates, relationships, and activities.
    Extracts structured triples from each dialogue line with multiple patterns.
    """
    # Relation patterns in priority order (most specific first)
    _VERB_PATTERNS = [
        # Ownership / profession
        (r"(?:i|i'm|my name is)\s+(\w+)", "speaker_name", None),
        (r"(?:i|i'm)\s+(?:a|an)\s+(.+?)(?:\.|,|$)", "occupation", None),
        (r"(?:i|i'm)\s+(?:working|employed)\s+(?:at|for|with)\s+(.+?)(?:\.|,|$)", "works_at", None),
        (r"(?:i|i've)\s+(?:moved|relocated|moved back)\s+to\s+(.+?)(?:\.|,|$)", "moved_to", None),
        (r"(?:i|i'm)\s+(?:living|staying|living now)\s+(?:in|at)\s+(.+?)(?:\.|,|$)", "lives_in", None),
        # Activity patterns
        (r"(?:i|we)\s+(went|traveled|flew|drove|visited|attended|joined)\s+(?:to\s+)?(.+?)(?:\.|,|$)", "visited", None),
        (r"(?:i|we)\s+(bought|purchased|got|received|ordered)\s+(.+?)(?:\.|,|$)", "acquired", None),
        (r"(?:i|we)\s+(started|began|tried|took up)\s+(.+?)(?:\.|,|$)", "started", None),
        (r"(?:i|we)\s+(stopped|quit|gave up|ended)\s+(.+?)(?:\.|,|$)", "stopped", None),
        (r"(?:i|we)\s+(like|love|enjoy|prefer|hate|dislike)\s+(.+?)(?:\.|,|$)", "preference", None),
        # Factual statements
        (r"(?:i|my)\s+(\w+)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)", "has_property", None),
        (r"(.+?)\s+(?:is|was|are|were)\s+(?:called|named)\s+(.+?)(?:\.|,|$)", "named", None),
        (r"(.+?)\s+(?:is|was|are|were)\s+(?:a|an|the)\s+(.+?)(?:\.|,|$)", "is_a", None),
        (r"(.+?)\s+(?:happens?|occurred?|took place)\s+(?:on|at|in)\s+(.+?)(?:\.|,|$)", "occurred_at", None),
        # Date/time patterns
        (r"(?:on|at|in)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d+(?:st|nd|rd|th)?)", "date", None),
        (r"(?:on|at)\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s*(?:\d{4})?)", "date", None),
    ]

    triples = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("---"):
            continue

        # Parse "[D1:3] Speaker: text"
        m = re.match(r"\[(D\d+:\d+)\]\s*([^:]+):\s*(.+)", line)
        if not m:
            continue
        dia_id, speaker, content = m.group(1), m.group(2).strip(), m.group(3).strip()
        if not content or len(content) < 5:
            continue

        # Use speaker_a's name directly as subject (avoid generic "User")
        subj = speaker

        normalized_date = _normalize_relative_date(content, session_date)
        if normalized_date:
            triples.extend([
                [f"event:{dia_id}", "speaker", subj],
                [f"event:{dia_id}", "date", normalized_date],
                [f"event:{dia_id}", "description", content[:200]],
                [subj, "event_date", normalized_date],
            ])

        # Core verb-based relation patterns
        matched = False
        for pattern, pred, obj_override in _VERB_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                groups = match.groups()
                if obj_override:
                    obj = obj_override
                elif len(groups) >= 2:
                    obj = groups[-1].strip().rstrip(".,;!?")
                else:
                    obj = groups[0].strip().rstrip(".,;!?")
                if obj and len(obj) > 1:
                    triples.append([subj, pred, obj[:120]])
                    if normalized_date:
                        triples.append([subj, "event_date", normalized_date])
                    matched = True
                    break  # one structural triple per line

        # Always add a full-text triple for dense retrieval coverage
        clean_content = content[:200].rstrip(".,;!?")
        if speaker == speaker_a:
            triples.append([speaker_a, "said", clean_content])
        else:
            triples.append([speaker, "said", clean_content])

        # Entity linking: extract proper nouns (capitalized words not at sentence start)
        words = content.split()
        for i, word in enumerate(words[1:], 1):
            word_clean = word.rstrip(".,;!?\"'")
            if (word_clean and word_clean[0].isupper() and len(word_clean) > 2
                    and word_clean.lower() not in {"i", "the", "a", "an", "my", "we", "our", "you", "they"}):
                triples.append([subj, "mentioned", word_clean])

    return triples[:120]  # increased cap for better coverage


def clear_conversation(pipeline, conversation_id: str, tenant_id: str = "default_tenant", user_id: str = "default_user"):
    """Remove all records for a conversation."""
    with pipeline.store._lock:
        conn = pipeline.store._get_connection()
        cursor = conn.cursor()
        for table in ("turns", "triples", "entities", "embeddings", "turn_embeddings"):
            cursor.execute(
                f"DELETE FROM {table} WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?",
                (conversation_id, tenant_id, user_id),
            )
        conn.commit()
        conn.close()
    if hasattr(pipeline, "retriever") and pipeline.retriever is not None:
        pipeline.retriever.reset_index()


def seed_conversation(
    pipeline,
    conv_data: Dict[str, Any],
    conversation_id: Optional[str] = None,
    tenant_id: str = "default_tenant",
    user_id: str = "default_user",
    extract_triples: bool = True,
    turn_level: bool = True,
) -> str:
    """
    Seed a parsed LoCoMo conversation into CGM store.

    Args:
        pipeline: Initialized CGMPipeline
        conv_data: Output of locomo_load.parse_conversation()
        conversation_id: Override; defaults to sample_id
        turn_level: If True, store each dialog turn separately (recommended)

    Returns:
        conversation_id used
    """
    cid = conversation_id or conv_data["sample_id"]
    speaker_a = conv_data["speaker_a"]

    clear_conversation(pipeline, cid, tenant_id, user_id)

    if turn_level:
        session_dates = {
            session["session_idx"]: session.get("date_time", "")
            for session in conv_data.get("sessions", [])
        }
        for turn in conv_data["turns"]:
            text = turn["formatted_line"]
            if not text.strip():
                continue

            session_date = session_dates.get(turn["session_idx"], "")
            triples = _simple_extract_triples(text, speaker_a, session_date) if extract_triples else []

            hmo = HybridMemoryObject(
                turn_id=turn["turn_id"],
                timestamp=float(turn["session_idx"] * 3600 + turn["turn_id"]),
                summary=text[:200],
                user_text=turn.get("user_text") or turn["text"],
                assistant_text=turn.get("assistant_text") or turn["text"],
                triples=triples,
                dia_id=turn.get("dia_id"),
            )
            hmo.raw_embedding = pipeline.retriever.embed_text(text)
            pipeline.store.save_hmo(cid, hmo, tenant_id=tenant_id, user_id=user_id)
            pipeline.retriever.store_turn(
                conversation_id=cid,
                turn_id=turn["turn_id"],
                text=text,
                summary=hmo.summary,
                user_text=hmo.user_text,
                assistant_text=hmo.assistant_text,
                tenant_id=tenant_id,
                user_id=user_id,
                dia_id=turn.get("dia_id"),
            )
    else:
        # Session-level fallback (legacy behavior)
        for session in conv_data["sessions"]:
            text = session["formatted_text"]
            triples = _simple_extract_triples(text, speaker_a, session.get("date_time", "")) if extract_triples else []
            hmo = HybridMemoryObject(
                turn_id=session["session_idx"],
                timestamp=float(session["session_idx"] * 3600),
                summary=f"Session {session['session_idx']}",
                user_text=f"Session {session['session_idx']}",
                assistant_text=text,
                triples=triples,
            )
            hmo.raw_embedding = pipeline.retriever.embed_text(text)
            pipeline.store.save_hmo(cid, hmo, tenant_id=tenant_id, user_id=user_id)

    return cid


def load_extracted_graph(pipeline, graph_path: str, conversation_id: str = "locomo_conv_0") -> str:
    """Load pre-extracted graph JSON (legacy format from data/locomo_extracted_graph.json)."""
    import json
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    clear_conversation(pipeline, conversation_id)

    for session in graph_data["sessions"]:
        text = session["formatted_text"]
        hmo = HybridMemoryObject(
            turn_id=session["session_idx"],
            timestamp=float(session["session_idx"] * 3600),
            summary=f"Dialogue session {session['session_idx']}",
            user_text=f"Dialogue session {session['session_idx']}",
            assistant_text=text,
            dia_id=f"D{session['session_idx']}:1",
        )
        hmo.raw_embedding = pipeline.retriever.embed_text(text)
        pipeline.store.save_hmo(conversation_id, hmo)

    with pipeline.store._lock:
        conn = pipeline.store._get_connection()
        cursor = conn.cursor()
        for trip_item in graph_data["triples"]:
            trip = trip_item["triple"]
            cursor.execute(
                "INSERT INTO triples (conversation_id, turn_id, subject, predicate, object) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, trip_item["session_idx"], trip[0], trip[1], trip[2]),
            )
        conn.commit()
        conn.close()

    return conversation_id
