"""
Training data construction for the Memory Encoder Network (MEN).

Builds per-triple training examples with proper [enc(subj||pred); enc(obj)] encoding,
and splits them into disjoint train/eval sets for generalization measurement.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TripleTrainingSample:
    """A single training example derived from a knowledge graph triple."""
    triple: List[str]            # [subject, predicate, object]
    x_vector: np.ndarray         # 768-dim: concat(enc(subj||pred), enc(obj))
    prompt_text: str             # Question prompt for the frozen LLM
    target_text: str             # Expected factual answer
    triple_text: str             # Natural language form: "subject predicate object"


# Handcrafted QA pairs for each triple in the benchmark dataset.
# Each triple maps to a (prompt, target) pair for teacher-forced training.
TRIPLE_QA_MAP: Dict[str, Dict[str, Any]] = {
    "Project|uses|FastAPI": {
        "prompts": [
            "User: What backend web framework is used in the project?\nAssistant:",
            "User: What backend web framework did we decide to use for our API project?\nAssistant:",
            "User: Which web framework does the project use?\nAssistant:"
        ],
        "target": "The project uses FastAPI as its backend web framework."
    },
    "Database|connects_to|PostgreSQL": {
        "prompts": [
            "User: What relational database system is the project connected to?\nAssistant:",
            "User: What relational database system are we connecting to?\nAssistant:",
            "User: Which database connects to the system?\nAssistant:"
        ],
        "target": "The database connects to PostgreSQL as the relational database system."
    },
    "Database|runs_on|Port 8080": {
        "prompts": [
            "User: On which port is the database running?\nAssistant:",
            "User: On which port number is our database running?\nAssistant:",
            "User: What port does the database run on?\nAssistant:"
        ],
        "target": "The database runs on port 8080."
    },
    "Database|uses|asyncpg": {
        "prompts": [
            "User: What asynchronous database connector is used?\nAssistant:",
            "User: What asynchronous database connector package are we using?\nAssistant:",
            "User: Which asynchronous library does the database use?\nAssistant:"
        ],
        "target": "The database uses asyncpg for asynchronous connectivity."
    },
    "Project|uses|typing_extensions": {
        "prompts": [
            "User: What type hints library does the project use?\nAssistant:",
            "User: Which library is used for type hints in the project?\nAssistant:"
        ],
        "target": "The project uses typing_extensions for type hints."
    },
    "Project|uses|Pydantic v2": {
        "prompts": [
            "User: What data validation library is used for schemas?\nAssistant:",
            "User: Which validation library is used for schemas?\nAssistant:"
        ],
        "target": "The project uses Pydantic v2 for data schema validations."
    },
    "Database|has_table|user_profiles": {
        "prompts": [
            "User: What is the name of the database table for user data?\nAssistant:",
            "User: What is the name of the database table we established for profiles?\nAssistant:",
            "User: Which database table is configured for user profiles?\nAssistant:"
        ],
        "target": "The database has a table called user_profiles for storing user data."
    },
    "Project|formatted_by|Ruff": {
        "prompts": [
            "User: What code formatting and linting tool is used?\nAssistant:",
            "User: What tool are we using for formatting and linting our python code?\nAssistant:",
            "User: Which formatter and linter is used in this project?\nAssistant:"
        ],
        "target": "The project is formatted by Ruff for code formatting and linting."
    },
    "Ruff|has_setting|Line length 100": {
        "prompts": [
            "User: What is the maximum line length setting for code formatting?\nAssistant:",
            "User: What is the maximum line length setting we configured for code formatting?\nAssistant:",
            "User: What line length limit is configured for Ruff?\nAssistant:"
        ],
        "target": "Ruff has a max line length setting of 100 characters."
    },
}

# Deterministic train/eval split indices (by triple key order)
# Train: 6 triples, Eval: 3 triples (disjoint)
TRAIN_KEYS = [
    "Project|uses|FastAPI",
    "Database|connects_to|PostgreSQL",
    "Database|runs_on|Port 8080",
    "Database|uses|asyncpg",
    "Project|uses|Pydantic v2",
    "Project|formatted_by|Ruff",
]

EVAL_KEYS = [
    "Project|uses|typing_extensions",
    "Database|has_table|user_profiles",
    "Ruff|has_setting|Line length 100",
]


def encode_triple(triple: List[str], embed_fn) -> np.ndarray:
    """
    Encodes a triple into a 768-dim vector using the proper [enc(subj||pred); enc(obj)] structure.
    
    Args:
        triple: [subject, predicate, object]
        embed_fn: Function that takes a string and returns a 384-dim numpy vector.
    
    Returns:
        768-dim numpy vector: concat(enc(subj + " " + pred), enc(obj))
    """
    sp_text = f"{triple[0]} {triple[1]}"
    o_text = triple[2]
    sp_emb = embed_fn(sp_text)   # shape: (384,)
    o_emb = embed_fn(o_text)     # shape: (384,)
    return np.concatenate([sp_emb, o_emb])  # shape: (768,)


def build_training_samples(triples: List[List[str]], embed_fn) -> List[TripleTrainingSample]:
    """
    Converts raw triples into TripleTrainingSample objects with proper encoding and prompt augmentation.
    
    Args:
        triples: List of [subject, predicate, object] lists.
        embed_fn: Function that takes a string and returns a 384-dim numpy vector.
        
    Returns:
        List of TripleTrainingSample, one or more per triple that has a matching QA pair.
    """
    samples = []
    for triple in triples:
        key = f"{triple[0]}|{triple[1]}|{triple[2]}"
        if key not in TRIPLE_QA_MAP:
            continue
        
        qa = TRIPLE_QA_MAP[key]
        x_vec = encode_triple(triple, embed_fn)
        triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
        
        prompts = qa.get("prompts") or [qa.get("prompt")]
        for p in prompts:
            if not p:
                continue
            samples.append(TripleTrainingSample(
                triple=triple,
                x_vector=x_vec,
                prompt_text=p,
                target_text=qa["target"],
                triple_text=triple_text,
            ))
    
    return samples


def split_train_eval(
    samples: List[TripleTrainingSample],
) -> Tuple[List[TripleTrainingSample], List[TripleTrainingSample]]:
    """
    Splits samples into disjoint train and eval sets using the deterministic key lists.
    
    Returns:
        (train_samples, eval_samples)
    """
    train_samples = []
    eval_samples = []
    
    for sample in samples:
        key = f"{sample.triple[0]}|{sample.triple[1]}|{sample.triple[2]}"
        if key in EVAL_KEYS:
            eval_samples.append(sample)
        elif key in TRAIN_KEYS:
            train_samples.append(sample)
        else:
            # Unknown triples default to train set
            train_samples.append(sample)
    
    return train_samples, eval_samples
