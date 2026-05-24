import os
import json
from phase1_validation.config import GeminiClient
from phase1_validation.extractor import GraphExtractor
from phase1_validation.simulator import DialogueSimulator, SCENARIOS
from phase1_validation.evaluator import DialogueEvaluator

def run_validation():
    print("=" * 60)
    print("      CONVERSATIONAL GRAPH MEMORY (CGM) - PHASE 1 VALIDATION")
    print("=" * 60)
    
    # 1. Initialize Client & Helpers
    print("[1/5] Initializing Gemini Client and Agents...")
    client = GeminiClient()
    extractor = GraphExtractor(client)
    simulator = DialogueSimulator(client)
    evaluator = DialogueEvaluator(client)
    
    results = []
    
    # 2. Run Scenarios
    for idx, scenario in enumerate(SCENARIOS, 1):
        print(f"\n[2/5] Running Scenario {idx}: '{scenario['name']}'")
        print(f"      Description: {scenario['description']}")
        
        # A. Format and show history
        history_text = simulator.format_history_as_text(scenario["history"])
        print(f"      History length: {len(history_text)} characters.")
        
        # B. Extract semantic memory (triples + summary)
        print("      Extracting semantic memory from history (CGM compression boundary)...")
        memory_data = extractor.extract(history_text)
        
        print("\n      --- Extracted Memory (HMO Preview) ---")
        print(f"      Summary: {memory_data.get('summary')}")
        print("      Semantic Triples:")
        for trip in memory_data.get("triples", [])[:5]:
            print(f"        - {trip}")
        if len(memory_data.get("triples", [])) > 5:
            print(f"        - ... and {len(memory_data.get('triples')) - 5} more.")
        print("      --------------------------------------\n")
        
        # C. Format memory as prompt context
        memory_prompt = extractor.format_memory_as_prompt(memory_data)
        
        # D. Run Baseline (with full history)
        print("      Running BASELINE generation (entire history in context)...")
        baseline_response = simulator.run_baseline(scenario)
        
        # E. Run CGM (with compressed memory prompt)
        print("      Running CGM generation (compressed graph/summary prompt)...")
        cgm_response = simulator.run_cgm(scenario, memory_prompt)
        
        # F. Grade and Evaluate
        print("      Evaluating and grading responses...")
        report = evaluator.evaluate_scenario(scenario, baseline_response, cgm_response)
        
        # G. Save transcripts and responses for report
        report["baseline_response"] = baseline_response
        report["cgm_response"] = cgm_response
        report["extracted_memory"] = memory_data
        
        results.append(report)
        
        # H. Print Scenario results summary
        print("-" * 50)
        print(f"      Scenario '{scenario['name']}' Results:")
        print(f"        Baseline Fact Recall Accuracy : {report['baseline_score']*100:.1f}%")
        print(f"        CGM Fact Recall Accuracy      : {report['cgm_score']*100:.1f}%")
        print(f"        Factual Fidelity (F_f)        : {report['factual_fidelity']*100:.1f}%")
        print(f"        (Facts retained: {report['retained_count']} / {report['baseline_recalled_count']} baseline recalled)")
        print("-" * 50)

    # 3. Aggregate results and output report
    print("\n[3/5] Aggregating results across all scenarios...")
    
    total_fidelity = sum(r["factual_fidelity"] for r in results)
    avg_fidelity = total_fidelity / len(results) if results else 0
    
    total_baseline_score = sum(r["baseline_score"] for r in results)
    avg_baseline_score = total_baseline_score / len(results) if results else 0
    
    total_cgm_score = sum(r["cgm_score"] for r in results)
    avg_cgm_score = total_cgm_score / len(results) if results else 0
    
    # 4. Save report.json
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(output_dir, "report.json")
    
    final_output = {
        "average_factual_fidelity": avg_fidelity,
        "average_baseline_recall": avg_baseline_score,
        "average_cgm_recall": avg_cgm_score,
        "scenarios": results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"[4/5] Saved complete test execution report to: {report_path}")
    
    # 5. Conclusion
    print("\n[5/5] CONCLUSION AND ANALYSIS:")
    print("=" * 60)
    print(f"Average Factual Recall (Baseline) : {avg_baseline_score*100:.1f}%")
    print(f"Average Factual Recall (CGM)      : {avg_cgm_score*100:.1f}%")
    print(f"Average Factual Fidelity (F_f)    : {avg_fidelity*100:.1f}%")
    print("-" * 60)
    
    target_threshold = 0.85
    if avg_fidelity >= target_threshold:
        print(f"SUCCESS: Average Factual Fidelity ({avg_fidelity*100:.1f}%) matches or exceeds the target threshold ({target_threshold*100:.1f}%).")
        print("Hypothesis H1 (Factual Fidelity Hypothesis) is VALIDATED!")
        print("The semantic graph representation preserves enough information for coherent context retrieval.")
    else:
        print(f"WARNING: Average Factual Fidelity ({avg_fidelity*100:.1f}%) is below the target threshold ({target_threshold*100:.1f}%).")
        print("Hypothesis H1 is NOT fully validated under text injection.")
        print("Recommendation: Consider fine-tuning adapters or testing a larger model for extraction/reconstruction.")
    print("=" * 60)

if __name__ == "__main__":
    run_validation()
