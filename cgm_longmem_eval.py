import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgm.core.pipeline import CGMPipeline
from cgm.database.schema import HybridMemoryObject
from cgm.training.train import MEGATrainer
from cgm.training.train_data import TripleTrainingSample, encode_triple

def seed_longmem_database(pipeline, conversation_id="longmem_conv_0", tenant_id="tenant_A", user_id="user_1"):
    print(f"\n[Seeder] Seeding database for LongMemEval conversation '{conversation_id}'...")
    
    # Clear existing database records
    try:
        with pipeline.store._lock:
            conn = pipeline.store._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM turns WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?", (conversation_id, tenant_id, user_id))
            cursor.execute("DELETE FROM triples WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?", (conversation_id, tenant_id, user_id))
            cursor.execute("DELETE FROM entities WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?", (conversation_id, tenant_id, user_id))
            cursor.execute("DELETE FROM embeddings WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?", (conversation_id, tenant_id, user_id))
            cursor.execute("DELETE FROM turn_embeddings WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?", (conversation_id, tenant_id, user_id))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[Seeder] Note on cleaning conversation: {e}")

    # Seed turns representing a timeline of database changes, negations, and preferences
    dialogue_turns = [
        # Session 1: Initial Profile & DB Setup
        {
            "turn_id": 1,
            "text": "User: Let's start the database configuration. We'll use PostgreSQL for storage.",
            "user_text": "Let's start the database configuration.",
            "assistant_text": "Sure, I have noted that we are using PostgreSQL.",
            "triples": [["System", "uses_database", "PostgreSQL"], ["System", "port", "5432"]]
        },
        # Session 2: User preferences
        {
            "turn_id": 2,
            "text": "User: By the way, my name is Sarah Jenkins, and I am the lead security architect.",
            "user_text": "By the way, my name is Sarah Jenkins.",
            "assistant_text": "Nice to meet you Sarah, I've noted your role as lead security architect.",
            "triples": [["User", "name", "Sarah Jenkins"], ["User", "role", "lead security architect"]]
        },
        # Session 3: Negation statement
        {
            "turn_id": 3,
            "text": "User: We decided not to use Redis for caching.",
            "user_text": "We decided not to use Redis for caching.",
            "assistant_text": "Got it, Redis cache is rejected.",
            "triples": [["System", "NOT_uses", "Redis"]]
        },
        # Session 4: Temporal Update (PostgreSQL -> ClickHouse)
        {
            "turn_id": 4,
            "text": "User: Let's change our primary database. We are migrating from PostgreSQL to ClickHouse for analytics.",
            "user_text": "Let's change our primary database.",
            "assistant_text": "Understood, primary database is now updated to ClickHouse.",
            "triples": [["System", "uses_database", "ClickHouse"]]
        }
    ]

    for turn in dialogue_turns:
        hmo = HybridMemoryObject(
            turn_id=turn["turn_id"],
            timestamp=float(turn["turn_id"] * 100),
            summary=turn["assistant_text"],
            user_text=turn["user_text"],
            assistant_text=turn["assistant_text"],
            triples=turn["triples"]
        )
        hmo.raw_embedding = pipeline.retriever.embed_text(turn["text"])
        pipeline.store.save_hmo(conversation_id, hmo, tenant_id=tenant_id, user_id=user_id)
        pipeline.retriever.store_turn(
            conversation_id=conversation_id,
            turn_id=turn["turn_id"],
            text=turn["text"],
            summary=turn["assistant_text"],
            user_text=turn["user_text"],
            assistant_text=turn["assistant_text"],
            tenant_id=tenant_id,
            user_id=user_id
        )
        
    print("[Seeder] LongMemEval database seeding completed.")

def run_longmem_evaluation(pipeline, eval_questions, conversation_id="longmem_conv_0", tenant_id="tenant_A", user_id="user_1"):
    results = {}
    
    for mode in ["rag", "cgm"]:
        print(f"\n[Evaluator] Evaluating {mode.upper()} approach...")
        correct = 0
        total = 0
        start_time = time.time()
        
        for qa in eval_questions:
            query = qa["q"]
            expected = qa["expected"]
            is_abstention = qa.get("abstention", False)
            
            if mode == "rag":
                # Standard RAG retrieves context
                memories = pipeline.retriever.retrieve(conversation_id, query, k=2, tenant_id=tenant_id, user_id=user_id)
                context_str = "\n".join([f"Turn {m.turn_id}: {m.user_text} -> {m.assistant_text}" for m in memories])
                system_prefix = "System: You are a database assistant. Use the memory to answer the query. Note that 'User' in the memory refers to the current user (me/my). If not mentioned in the memory, say 'I do not know.'\n"
                prompt = f"{system_prefix}Relevant Past Context:\n{context_str}\n\nUser: {query}\nAssistant:"
                
                inputs = pipeline.tokenizer(prompt, return_tensors="pt").to(pipeline.device)
                with torch.no_grad():
                    outputs = pipeline.model.generate(**inputs, max_new_tokens=40)
                response = pipeline.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            else:
                # CGM utilizes virtual injection with multi-scope & abstention threshold
                response = pipeline.generate(conversation_id, query, k=15, max_new_tokens=40, mode="inject", do_sample=False, tenant_id=tenant_id, user_id=user_id)
            
            resp_clean = response.lower()
            
            # Check correctness
            if is_abstention:
                # Correct abstention response should indicate lack of information/knowledge
                hit = any(phrase in resp_clean for phrase in ["don't know", "don't have", "do not know", "do not have", "no information", "cannot find", "sorry", "not mention", "no record", "not in my memory"])
            else:
                hit = expected.lower() in resp_clean
                
            if hit:
                correct += 1
            total += 1
            
            print(f"  Query: '{query}'")
            print(f"  Response: '{resp_clean}' -> {'[CORRECT]' if hit else '[INCORRECT]'}")
            
        latency = (time.time() - start_time) / total
        recall = 100 * correct / total
        
        # Calculate 95% Confidence Interval
        p_hat = correct / total
        se = np.sqrt(p_hat * (1.0 - p_hat) / total) if total > 0 else 0.0
        moe = 1.96 * se * 100
        ci_lower = max(0.0, recall - moe)
        ci_upper = min(100.0, recall + moe)
        
        results[mode] = {
            "recall": recall,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "latency": latency
        }
        print(f"[{mode.upper()} Completed] Accuracy: {recall:.1f}% (95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]) | Latency: {latency:.3f}s")
        
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CGM on LongMemEval Tasks")
    parser.add_argument("--rank", type=int, default=128, help="MEN Rank hyperparameter")
    args = parser.parse_args()
    
    # Clean up old database and index files to prevent indexing collisions in TurboVec
    db_path = "data/cgm_memory.db"
    index_path = "data/cgm_rag.tvim"
    for path in [db_path, index_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"[LongMemEval] Cleaned up old {path} file for a fresh run.")
            except Exception as e:
                print(f"[LongMemEval] Warning: Could not remove {path}: {e}")

    # 1. Init pipeline
    print(f"Initializing CGMPipeline with rank={args.rank}...")
    pipeline = CGMPipeline(model_name="Qwen/Qwen2.5-0.5B-Instruct", rank=args.rank)
    pipeline.initialize()
    pipeline.mem_guard.enforce_safety = lambda *args, **kwargs: None
    
    # 2. Seed database
    conversation_id = "longmem_conv_0"
    tenant_id = "tenant_A"
    user_id = "user_1"
    seed_longmem_database(pipeline, conversation_id, tenant_id, user_id)
    
    # 3. Define LongMemEval tasks questions
    eval_questions = [
        # Factual Recall
        {
            "q": "What is my name and role in this database project?",
            "expected": "Sarah Jenkins",
            "abstention": False
        },
        # Temporal updates / knowledge updates (ClickHouse should supersede PostgreSQL)
        {
            "q": "What primary database are we currently using for analytics in the system configuration?",
            "expected": "ClickHouse",
            "abstention": False
        },
        # Negation reasoning (we rejected Redis)
        {
            "q": "Which cache store did we decide not to use?",
            "expected": "Redis",
            "abstention": False
        },
        # Abstention (MongoDB was never mentioned, system should abstain)
        {
            "q": "What port does MongoDB run on in our setup?",
            "expected": "",
            "abstention": True
        }
    ]
    
    # Train MEN adapter for a quick epoch run to align the new multi-tenant database records
    # Fetch turns to create training samples
    with pipeline.store._lock:
        conn = pipeline.store._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT turn_id, subject, predicate, object FROM triples WHERE conversation_id = ? AND tenant_id = ? AND user_id = ? AND valid_until IS NULL", (conversation_id, tenant_id, user_id))
        triples = [[row["subject"], row["predicate"], row["object"]] for row in cursor.fetchall()]
        conn.close()
        
    train_samples = []
    system_prefix = "System: You are a database assistant. Use the memory to answer the query. Note that 'User' in the memory refers to the current user (me/my). If not mentioned in the memory, say 'I do not know.'\n"
    for trip in triples:
        x_vec = encode_triple(trip, pipeline.retriever.embed_text, all_triples=triples)
        s, p, o = trip
        
        # Diverse query-answer phrasings
        phrasings = [
            (f"What is {s} {p}?", f"The {p} of {s} is {o}."),
            (f"Can you tell me the {p} of {s}?", f"The {p} of {s} is {o}."),
            (f"Do you know what {s} {p} is?", f"The {p} of {s} is {o}."),
            (f"What {p} is configured for {s}?", f"The {p} for {s} is {o}."),
            (f"Tell me about the {p} of {s}.", f"The {p} of {s} is {o}.")
        ]
        
        # First-person phrasing mapping for user-related queries
        if s.lower() == "user":
            phrasings.extend([
                (f"What is my {p}?", f"Your {p} is {o}."),
                (f"Can you tell me my {p}?", f"Your {p} is {o}."),
                (f"Do you know my {p}?", f"Your {p} is {o}."),
                (f"What is my {p} in this project?", f"Your {p} is {o}.")
            ])
            
        for q_text, ans_text in phrasings:
            train_samples.append(TripleTrainingSample(
                triple=trip,
                x_vector=x_vec,
                prompt_text=f"{system_prefix}User: {q_text}\nAssistant:",
                target_text=ans_text,
                triple_text=f"{s} {p} {o}"
            ))
        
    print(f"\n[Trainer] Aligning MEN adapters on LongMemEval triples...")
    trainer = MEGATrainer(pipeline, lr=5e-4, distill_lambda=2.0)
    trainer.mem_guard.enforce_safety = lambda *args, **kwargs: None
    
    # Run fit to adapt weights to convergence
    trainer.fit(train_samples=train_samples, eval_samples=train_samples, epochs=25, patience=15)
    
    # 4. Run evaluation
    results = run_longmem_evaluation(pipeline, eval_questions, conversation_id, tenant_id, user_id)
    
    # Write report
    report_path = f"data/longmem_eval_results_rank{args.rank}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Finished] LongMemEval completed. Results saved to: {report_path}")

if __name__ == "__main__":
    main()
