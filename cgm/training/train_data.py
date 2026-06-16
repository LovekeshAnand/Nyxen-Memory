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
# Organized across 5 topic domains for statistical robustness (35+ total triples).
TRIPLE_QA_MAP: Dict[str, Dict[str, Any]] = {
    # ═══════════════════════════════════════════════════════════
    # DOMAIN 1: Backend API Setup (9 triples)
    # ═══════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════
    # DOMAIN 2: ML Model Training Configuration (7 triples)
    # ═══════════════════════════════════════════════════════════
    "Model|uses_optimizer|AdamW": {
        "prompts": [
            "User: What optimizer are we using for model training?\nAssistant:",
            "User: Which optimization algorithm is configured for the training run?\nAssistant:",
        ],
        "target": "The model uses AdamW as the optimizer for training."
    },
    "Training|has_learning_rate|3e-4": {
        "prompts": [
            "User: What learning rate did we set for training?\nAssistant:",
            "User: What is the configured learning rate for the model?\nAssistant:",
        ],
        "target": "The training learning rate is set to 3e-4."
    },
    "Training|has_batch_size|32": {
        "prompts": [
            "User: What batch size are we using for training?\nAssistant:",
            "User: What is the training batch size we configured?\nAssistant:",
        ],
        "target": "The training batch size is 32."
    },
    "Training|runs_on|A100 GPU": {
        "prompts": [
            "User: What GPU are we using for model training?\nAssistant:",
            "User: Which GPU hardware is the training running on?\nAssistant:",
        ],
        "target": "The training runs on an A100 GPU."
    },
    "Training|uses_dataset|ImageNet": {
        "prompts": [
            "User: What dataset are we training the model on?\nAssistant:",
            "User: Which dataset is used for the training run?\nAssistant:",
        ],
        "target": "The training uses the ImageNet dataset."
    },
    "Model|has_architecture|ResNet-50": {
        "prompts": [
            "User: What is the model architecture we selected?\nAssistant:",
            "User: Which neural network architecture are we training?\nAssistant:",
        ],
        "target": "The model architecture is ResNet-50."
    },
    "Training|has_epochs|100": {
        "prompts": [
            "User: How many epochs are we training for?\nAssistant:",
            "User: What is the total number of training epochs configured?\nAssistant:",
        ],
        "target": "The training is configured for 100 epochs."
    },

    # ═══════════════════════════════════════════════════════════
    # DOMAIN 3: DevOps & Deployment (7 triples)
    # ═══════════════════════════════════════════════════════════
    "App|deploys_to|AWS": {
        "prompts": [
            "User: What cloud platform are we deploying to?\nAssistant:",
            "User: Where is the application being deployed?\nAssistant:",
        ],
        "target": "The application deploys to AWS."
    },
    "Container|uses|Docker": {
        "prompts": [
            "User: What containerization tool are we using?\nAssistant:",
            "User: Which container runtime is configured for the deployment?\nAssistant:",
        ],
        "target": "The container uses Docker for containerization."
    },
    "CI|uses|GitHub Actions": {
        "prompts": [
            "User: What CI/CD platform are we using?\nAssistant:",
            "User: Which continuous integration system is set up?\nAssistant:",
        ],
        "target": "The CI pipeline uses GitHub Actions."
    },
    "Deployment|has_region|us-east-1": {
        "prompts": [
            "User: What AWS region are we deploying to?\nAssistant:",
            "User: Which cloud region is configured for our deployment?\nAssistant:",
        ],
        "target": "The deployment region is us-east-1."
    },
    "Deployment|has_instance_type|t3.medium": {
        "prompts": [
            "User: What EC2 instance type are we using?\nAssistant:",
            "User: Which instance type is configured for the server?\nAssistant:",
        ],
        "target": "The deployment uses t3.medium instance type."
    },
    "Monitoring|uses|Prometheus": {
        "prompts": [
            "User: What monitoring tool are we using?\nAssistant:",
            "User: Which system monitoring solution is configured?\nAssistant:",
        ],
        "target": "The monitoring system uses Prometheus."
    },
    "LoadBalancer|is_type|ALB": {
        "prompts": [
            "User: What type of load balancer are we using?\nAssistant:",
            "User: Which load balancer is configured for the deployment?\nAssistant:",
        ],
        "target": "The load balancer is an ALB (Application Load Balancer)."
    },

    # ═══════════════════════════════════════════════════════════
    # DOMAIN 4: Frontend Project Setup (7 triples)
    # ═══════════════════════════════════════════════════════════
    "Frontend|uses|React": {
        "prompts": [
            "User: What frontend framework are we using?\nAssistant:",
            "User: Which UI library is the frontend built with?\nAssistant:",
        ],
        "target": "The frontend uses React as its UI framework."
    },
    "StateManagement|uses|Zustand": {
        "prompts": [
            "User: What state management library are we using?\nAssistant:",
            "User: Which state management solution is configured for the frontend?\nAssistant:",
        ],
        "target": "The state management uses Zustand."
    },
    "Styling|uses|TailwindCSS": {
        "prompts": [
            "User: What CSS framework are we using for styling?\nAssistant:",
            "User: Which styling solution is configured for the frontend?\nAssistant:",
        ],
        "target": "The styling uses TailwindCSS."
    },
    "Testing|uses|Vitest": {
        "prompts": [
            "User: What testing framework are we using for the frontend?\nAssistant:",
            "User: Which test runner is configured for the project?\nAssistant:",
        ],
        "target": "The testing framework is Vitest."
    },
    "BuildTool|is|Vite": {
        "prompts": [
            "User: What build tool are we using for the frontend?\nAssistant:",
            "User: Which bundler is configured for the frontend project?\nAssistant:",
        ],
        "target": "The build tool is Vite."
    },
    "PackageManager|is|pnpm": {
        "prompts": [
            "User: What package manager are we using?\nAssistant:",
            "User: Which package manager is configured for the project?\nAssistant:",
        ],
        "target": "The package manager is pnpm."
    },
    "NodeVersion|is|20": {
        "prompts": [
            "User: What Node.js version are we using?\nAssistant:",
            "User: Which Node version is configured for the project?\nAssistant:",
        ],
        "target": "The Node.js version is 20."
    },

    # ═══════════════════════════════════════════════════════════
    # DOMAIN 5: Data Pipeline Configuration (7 triples)
    # ═══════════════════════════════════════════════════════════
    "Pipeline|uses|Apache Spark": {
        "prompts": [
            "User: What distributed processing framework are we using?\nAssistant:",
            "User: Which data processing engine is configured for the pipeline?\nAssistant:",
        ],
        "target": "The pipeline uses Apache Spark for distributed processing."
    },
    "Storage|uses|S3": {
        "prompts": [
            "User: What object storage are we using for the data pipeline?\nAssistant:",
            "User: Where is the pipeline data stored?\nAssistant:",
        ],
        "target": "The storage uses S3 for the data pipeline."
    },
    "DataFormat|is|Parquet": {
        "prompts": [
            "User: What file format are we using for the data?\nAssistant:",
            "User: Which data format is configured for storage?\nAssistant:",
        ],
        "target": "The data format is Parquet."
    },
    "Scheduler|uses|Airflow": {
        "prompts": [
            "User: What workflow scheduler are we using?\nAssistant:",
            "User: Which orchestration tool schedules the pipeline?\nAssistant:",
        ],
        "target": "The scheduler uses Airflow for workflow orchestration."
    },
    "AnalyticsDB|is|DuckDB": {
        "prompts": [
            "User: What analytics database are we using?\nAssistant:",
            "User: Which analytical query engine is configured?\nAssistant:",
        ],
        "target": "The analytics database is DuckDB."
    },
    "Partitioning|has_key|date": {
        "prompts": [
            "User: What is the partitioning key for the data?\nAssistant:",
            "User: Which column is used as the partition key?\nAssistant:",
        ],
        "target": "The partitioning key is date."
    },
    "Compression|uses|Snappy": {
        "prompts": [
            "User: What compression codec are we using for the data?\nAssistant:",
            "User: Which compression algorithm is configured for the pipeline?\nAssistant:",
        ],
        "target": "The compression codec is Snappy."
    },
    # ═══════════════════════════════════════════════════════════
    # DOMAIN 6: State & Goals (4 triples)
    # ═══════════════════════════════════════════════════════════
    "CurrentTask|aims_to|fix_cache_collision": {
        "prompts": [
            "User: What does the current task aim to do?\nAssistant:",
            "User: What is the goal of our current active task?\nAssistant:"
        ],
        "target": "The current task aims to fix cache collision."
    },
    "DecidedSetup|uses_package|typing_extensions": {
        "prompts": [
            "User: What package does the decided setup use?\nAssistant:",
            "User: Which library did we decide to use in our setup?\nAssistant:"
        ],
        "target": "The decided setup uses package typing_extensions."
    },
    "SystemState|has_status|degraded_performance": {
        "prompts": [
            "User: What is the current status of the system state?\nAssistant:",
            "User: How is the system state currently performing?\nAssistant:"
        ],
        "target": "The system state has status degraded_performance."
    },
    "DevelopmentGoal|targets_platform|kubernetes": {
        "prompts": [
            "User: What platform does our development goal target?\nAssistant:",
            "User: Which deployment platform are we targeting for development?\nAssistant:"
        ],
        "target": "The development goal targets platform kubernetes."
    },
}

# ═══════════════════════════════════════════════════════════
# Deterministic train/eval split (70/30 by domain)
# Each domain contributes ~5 train + ~2 eval triples
# ═══════════════════════════════════════════════════════════
TRAIN_KEYS = [
    # Domain 1: Backend (6 train)
    "Project|uses|FastAPI",
    "Database|connects_to|PostgreSQL",
    "Database|runs_on|Port 8080",
    "Database|uses|asyncpg",
    "Project|uses|Pydantic v2",
    "Project|formatted_by|Ruff",
    # Domain 2: ML Training (5 train)
    "Model|uses_optimizer|AdamW",
    "Training|has_learning_rate|3e-4",
    "Training|has_batch_size|32",
    "Training|runs_on|A100 GPU",
    "Training|uses_dataset|ImageNet",
    # Domain 3: DevOps (5 train)
    "App|deploys_to|AWS",
    "Container|uses|Docker",
    "CI|uses|GitHub Actions",
    "Deployment|has_region|us-east-1",
    "Deployment|has_instance_type|t3.medium",
    # Domain 4: Frontend (5 train)
    "Frontend|uses|React",
    "StateManagement|uses|Zustand",
    "Styling|uses|TailwindCSS",
    "Testing|uses|Vitest",
    "BuildTool|is|Vite",
    # Domain 5: Data Pipeline (5 train)
    "Pipeline|uses|Apache Spark",
    "Storage|uses|S3",
    "DataFormat|is|Parquet",
    "Scheduler|uses|Airflow",
    "AnalyticsDB|is|DuckDB",
    # Domain 6: State & Goals (2 train)
    "CurrentTask|aims_to|fix_cache_collision",
    "DecidedSetup|uses_package|typing_extensions",
]

EVAL_KEYS = [
    # Domain 1: Backend (3 eval)
    "Project|uses|typing_extensions",
    "Database|has_table|user_profiles",
    "Ruff|has_setting|Line length 100",
    # Domain 2: ML Training (2 eval)
    "Model|has_architecture|ResNet-50",
    "Training|has_epochs|100",
    # Domain 3: DevOps (2 eval)
    "Monitoring|uses|Prometheus",
    "LoadBalancer|is_type|ALB",
    # Domain 4: Frontend (2 eval)
    "PackageManager|is|pnpm",
    "NodeVersion|is|20",
    # Domain 5: Data Pipeline (2 eval)
    "Partitioning|has_key|date",
    "Compression|uses|Snappy",
    # Domain 6: State & Goals (2 eval)
    "SystemState|has_status|degraded_performance",
    "DevelopmentGoal|targets_platform|kubernetes",
]


def encode_triple(triple: List[str], embed_fn, all_triples: Optional[List[List[str]]] = None) -> np.ndarray:
    """
    Encodes a triple into a 768-dim vector using the proper [enc(subj||pred); enc(obj)] structure.
    If all_triples is provided, it searches for 2-hop connected subgraphs and encodes them as a relational path.
    
    Args:
        triple: [subject, predicate, object]
        embed_fn: Function that takes a string and returns a 384-dim numpy vector.
        all_triples: Optional list of all triples in the graph to find connected paths.
    
    Returns:
        768-dim numpy vector: concat(enc(left_text), enc(right_text))
    """
    s, p, o = triple
    left_text = f"{s} {p}"
    right_text = o
    
    if all_triples:
        # 1. Forward connection: find another triple starting with our object
        forward_conn = None
        for ot in all_triples:
            if ot != triple and len(ot) >= 3 and ot[0].lower() == o.lower():
                forward_conn = ot
                break
        
        if forward_conn:
            left_text = f"{s} {p} {o} {forward_conn[1]}"
            right_text = forward_conn[2]
        else:
            # 2. Backward connection: find another triple ending with our subject
            backward_conn = None
            for ot in all_triples:
                if ot != triple and len(ot) >= 3 and ot[2].lower() == s.lower():
                    backward_conn = ot
                    break
            
            if backward_conn:
                left_text = f"{backward_conn[0]} {backward_conn[1]} {s} {p}"
                right_text = o
                
    sp_emb = embed_fn(left_text)   # shape: (384,)
    o_emb = embed_fn(right_text)     # shape: (384,)
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
        x_vec = encode_triple(triple, embed_fn, all_triples=triples)
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
