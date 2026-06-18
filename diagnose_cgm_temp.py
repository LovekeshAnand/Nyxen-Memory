import sys
import os
import torch
import numpy as np

# Clean up files
db_path = "data/cgm_memory.db"
index_path = "data/cgm_rag.tvim"
for path in [db_path, index_path]:
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

root_dir = r"d:\Nyxen-Memory"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cgm.core.pipeline import CGMPipeline
from cgm.training.train import MEGATrainer
from cgm.training.train_data import TripleTrainingSample, encode_triple
from transformers.cache_utils import DynamicCache

pipeline = CGMPipeline(model_name="Qwen/Qwen2.5-0.5B-Instruct")
pipeline.mem_guard.enforce_safety = lambda *args, **kwargs: None
pipeline.initialize(verbose=False)

# Seed database and fetch active triples
conversation_id = "longmem_conv_0"
tenant_id = "tenant_A"
user_id = "user_1"

# Clear & Seed
from cgm_longmem_eval import seed_longmem_database, run_longmem_evaluation
seed_longmem_database(pipeline, conversation_id, tenant_id, user_id)

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
    phrasings = [
        (f"What is {s} {p}?", f"The {p} of {s} is {o}."),
        (f"Can you tell me the {p} of {s}?", f"The {p} of {s} is {o}."),
        (f"Do you know what {s} {p} is?", f"The {p} of {s} is {o}."),
        (f"What {p} is configured for {s}?", f"The {p} for {s} is {o}."),
        (f"Tell me about the {p} of {s}.", f"The {p} of {s} is {o}.")
    ]
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

print(f"Total triples: {len(triples)} | Augmented samples: {len(train_samples)}")

# Use distill_lambda = 2.0 to prioritize KV-distillation
trainer = MEGATrainer(pipeline, lr=5e-4, distill_lambda=2.0)
trainer.mem_guard.enforce_safety = lambda *args, **kwargs: None

# Train to convergence (epochs=100)
trainer.fit(train_samples=train_samples, eval_samples=train_samples, epochs=100, patience=50)

# Build static x_all_tensor
unique_x_vectors = []
seen = set()
for s in train_samples:
    t_key = f"{s.triple[0]}|{s.triple[1]}|{s.triple[2]}"
    if t_key not in seen:
        seen.add(t_key)
        unique_x_vectors.append(s.x_vector)
x_all = np.stack(unique_x_vectors)
x_all_tensor = torch.tensor(x_all, dtype=torch.float32).unsqueeze(0).to(pipeline.device)

# Diagnose Query 1 using trainer style
print("\n=== DIAGNOSING QUERY 1 (SARAH JENKINS) ===")
# Find a sample that queries name
sample = None
for s in train_samples:
    if s.triple[0] == "User" and s.triple[1] == "name" and "User: What is my name?" in s.prompt_text:
        sample = s
        break
if sample is None:
    sample = train_samples[0]
print("Prompt Text:", sample.prompt_text)
print("Expected Output:", sample.target_text)

# Run trainer evaluate_recall style
pipeline.men.eval()
with torch.no_grad():
    raw_kv = pipeline.men(x_all_tensor)
    past_kv = DynamicCache()
    attn_dtype = next(pipeline.model.parameters()).dtype
    
    memory_length = x_all_tensor.shape[1]
    pos_ids = torch.arange(memory_length, dtype=torch.long, device=pipeline.device).unsqueeze(0)
    dummy_x = torch.zeros(1, 1, memory_length, pipeline.model.config.hidden_size // pipeline.model.config.num_attention_heads).to(pipeline.device)
    cos, sin = pipeline.model.model.rotary_emb(dummy_x, pos_ids)
    cos_u = cos.unsqueeze(1).to(dtype=attn_dtype)
    sin_u = sin.unsqueeze(1).to(dtype=attn_dtype)
    
    for idx, (k, v) in enumerate(raw_kv):
        k_cast = k.to(dtype=attn_dtype)
        v_cast = v.to(dtype=attn_dtype)
        k1 = k_cast[..., : k_cast.shape[-1] // 2]
        k2 = k_cast[..., k_cast.shape[-1] // 2 :]
        rot_k = torch.cat((-k2, k1), dim=-1)
        k_cast = (k_cast * cos_u) + (rot_k * sin_u)
        past_kv.update(k_cast, v_cast, idx)

    # Decode with trainer query
    prompt_ids = pipeline.tokenizer(sample.prompt_text, return_tensors="pt").input_ids.to(pipeline.device)
    memory_mask = torch.ones(1, memory_length, dtype=torch.long, device=pipeline.device)
    prompt_mask = torch.ones_like(prompt_ids)
    full_mask = torch.cat([memory_mask, prompt_mask], dim=-1)
    
    position_ids = torch.arange(memory_length, memory_length + prompt_ids.shape[1], dtype=torch.long, device=pipeline.device).unsqueeze(0)
    
    outputs = pipeline.model.generate(
        prompt_ids,
        past_key_values=past_kv,
        attention_mask=full_mask,
        position_ids=position_ids,
        max_new_tokens=30,
        pad_token_id=pipeline.tokenizer.pad_token_id,
        do_sample=False,
    )
    resp_trainer = pipeline.tokenizer.decode(outputs[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
    print("Trainer Style Response:", resp_trainer)

# Run pipeline style
eval_q = "What is my name and role in this database project?"
print("\nPipeline Style Query:", eval_q)
resp_pipeline = pipeline.generate(conversation_id, eval_q, k=15, max_new_tokens=30, mode="inject", do_sample=False, tenant_id=tenant_id, user_id=user_id)
print("Pipeline Style Response:", resp_pipeline)

# Run pipeline style but with training prompt!
print("\nPipeline Style Query (with training prompt):", sample.prompt_text.replace("User:", "").replace("Assistant:", "").strip())
resp_pipeline_train_prompt = pipeline.generate(
    conversation_id, 
    sample.prompt_text.replace("User:", "").replace("Assistant:", "").strip(), 
    k=15, max_new_tokens=30, mode="inject", do_sample=False, tenant_id=tenant_id, user_id=user_id
)
print("Pipeline Style Response (training prompt):", resp_pipeline_train_prompt)
