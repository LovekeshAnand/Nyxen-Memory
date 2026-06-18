"""Ingest LongMemEval haystacks into CGM store (one conversation per question)."""
from typing import Any, Dict, List

from cgm.database.schema import HybridMemoryObject
from cgm.eval.locomo_ingest import clear_conversation, _simple_extract_triples


def seed_haystack(
    pipeline,
    instance: Dict[str, Any],
    tenant_id: str = "default_tenant",
    user_id: str = "default_user",
    extract_triples: bool = True,
) -> str:
    """
    Ingest one LongMemEval instance's haystack sessions as turns.

    Returns conversation_id (= question_id).
    """
    cid = instance["question_id"]
    clear_conversation(pipeline, cid, tenant_id, user_id)

    turn_id = 0
    haystack_sessions = instance.get("haystack_sessions", [])
    haystack_dates = instance.get("haystack_dates", [])

    for sess_idx, session in enumerate(haystack_sessions):
        date_str = haystack_dates[sess_idx] if sess_idx < len(haystack_dates) else ""
        for turn in session:
            turn_id += 1
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content.strip():
                continue

            text = f"[{role}] {content}"
            triples = []
            if extract_triples:
                triples = _simple_extract_triples(text, "user")
                if role == "user":
                    triples.append(["User", "said", content[:120]])
                else:
                    triples.append(["Assistant", "said", content[:120]])

            hmo = HybridMemoryObject(
                turn_id=turn_id,
                timestamp=float(sess_idx * 1000 + turn_id),
                summary=content[:200],
                user_text=content if role == "user" else "",
                assistant_text=content if role == "assistant" else "",
                triples=triples,
            )
            hmo.raw_embedding = pipeline.retriever.embed_text(text)
            pipeline.store.save_hmo(cid, hmo, tenant_id=tenant_id, user_id=user_id)
            pipeline.retriever.store_turn(
                conversation_id=cid,
                turn_id=turn_id,
                text=text,
                summary=hmo.summary,
                user_text=hmo.user_text,
                assistant_text=hmo.assistant_text,
                tenant_id=tenant_id,
                user_id=user_id,
            )

    return cid
