"""
Official LoCoMo F1 scoring (ported from snap-research/locomo task_eval/evaluation.py).
Lightweight: uses stdlib + simple stemming; no bert_score dependency.
"""
import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List, Any, Optional

import numpy as np

# Porter stemmer suffix rules (minimal subset for benchmark parity)
_STEM_SUFFIXES = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)


def _stem_word(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, replacement in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s, flags=re.IGNORECASE)
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return s.lower()


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [_stem_word(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [_stem_word(w) for w in normalize_answer(ground_truth).split()]
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_answer(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(np.mean([
        max(f1_score(pred, gt) for pred in predictions)
        for gt in ground_truths
    ]))


def score_qa_item(item: Dict[str, Any], prediction: str) -> float:
    """
    Score a single LoCoMo QA item with official category rules.
    item must have 'category' and 'answer' (or adversarial_answer for cat 5).
    """
    category = int(item["category"])
    answer = item.get("answer") or item.get("adversarial_answer") or ""
    answer = str(answer)

    if category == 3:
        answer = answer.split(";")[0].strip()

    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 1:
        return f1_multi_answer(prediction, answer)
    if category == 5:
        lower = prediction.lower()
        if "no information available" in lower or "not mentioned" in lower:
            return 1.0
        return 0.0
    raise ValueError(f"Unknown category: {category}")


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def aggregate_results(
    results: List[Dict[str, Any]],
    score_key: str = "f1",
) -> Dict[str, Any]:
    """Aggregate per-item F1 scores into overall and per-category means."""
    by_category: Dict[int, List[float]] = {}
    all_scores: List[float] = []

    for row in results:
        f1 = row.get(score_key, 0.0)
        cat = int(row.get("category", 0))
        all_scores.append(f1)
        by_category.setdefault(cat, []).append(f1)

    per_category = {
        CATEGORY_NAMES.get(cat, str(cat)): round(float(np.mean(scores)) * 100, 2)
        for cat, scores in sorted(by_category.items())
    }

    return {
        "overall_f1": round(float(np.mean(all_scores)) * 100, 2) if all_scores else 0.0,
        "num_questions": len(all_scores),
        "per_category": per_category,
    }


def is_abstention_correct(prediction: str) -> bool:
    lower = prediction.lower()
    return any(p in lower for p in (
        "no information available", "not mentioned", "i do not know",
        "don't know", "do not know", "no information",
    ))
