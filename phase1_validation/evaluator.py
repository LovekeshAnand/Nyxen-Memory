import json
from typing import Dict, Any, List
from phase1_validation.config import GeminiClient

GRADER_SYSTEM_INSTRUCTION = """You are a strict and objective AI Grader.
Your job is to evaluate an AI response against a list of factual criteria and check if each criterion is satisfied.

For each criterion in the list:
- "satisfied" should be true if the AI response correctly mentions, configures, or uses the specific technology, library, setting, port number, table name, or column structure requested.
- "satisfied" should be false if the AI response omits, contradicts, or gets the detail wrong.
- Provide a brief "reason" (1 sentence) explaining why it is satisfied or not.

You must output a JSON object matching this schema:
{
  "results": [
    {
      "criterion": "String representing the criterion",
      "satisfied": true,
      "reason": "Explanation of the grade"
    }
  ]
}

Only return valid JSON. Do not write any explanations outside the JSON."""

class DialogueEvaluator:
    """
    Grades the generated responses using Gemini as an objective evaluator
    and computes the relative Factual Fidelity (F_f).
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def grade_response(self, response_text: str, criteria: List[str], max_parse_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Calls Gemini to grade the response text against the list of criteria.
        Implements a retry loop for robust JSON parsing.
        """
        prompt = f"AI Response to Grade:\n{response_text}\n\nCriteria to Check:\n"
        for i, c in enumerate(criteria, 1):
            prompt += f"{i}. {c}\n"
            
        for attempt in range(max_parse_retries):
            try:
                raw_result = self.client.generate(
                    prompt=prompt,
                    system_instruction=GRADER_SYSTEM_INSTRUCTION,
                    json_mode=True
                )
                data = json.loads(raw_result.strip())
                return data.get("results", [])
            except json.JSONDecodeError as jde:
                print(f"      [WARNING] JSON decode failed during grading (Attempt {attempt+1}/{max_parse_retries}): {jde}")
                if attempt == max_parse_retries - 1:
                    break
            except Exception as e:
                print(f"      [ERROR] Grading attempt {attempt+1} failed with error: {e}")
                if attempt == max_parse_retries - 1:
                    break
                    
        # Fallback output
        return [{"criterion": c, "satisfied": False, "reason": "Grading failed due to parsing or API error."} for c in criteria]

    def evaluate_scenario(self, scenario: Dict[str, Any], baseline_resp: str, cgm_resp: str) -> Dict[str, Any]:
        """
        Runs grading for both baseline and CGM responses, then computes fidelity metrics.
        """
        criteria = scenario["criteria"]
        
        print(f"  Grading baseline response for '{scenario['name']}'...")
        baseline_grades = self.grade_response(baseline_resp, criteria)
        
        print(f"  Grading CGM response for '{scenario['name']}'...")
        cgm_grades = self.grade_response(cgm_resp, criteria)
        
        # Calculate scores
        baseline_satisfied = sum(1 for g in baseline_grades if g["satisfied"])
        cgm_satisfied = sum(1 for g in cgm_grades if g["satisfied"])
        
        total_criteria = len(criteria)
        baseline_score = baseline_satisfied / total_criteria if total_criteria > 0 else 0
        cgm_score = cgm_satisfied / total_criteria if total_criteria > 0 else 0
        
        # Relative Fidelity (F_f):
        # Out of the requirements that the Baseline successfully recalled,
        # how many did the CGM run also successfully recall?
        baseline_satisfied_set = {g["criterion"] for g in baseline_grades if g["satisfied"]}
        cgm_satisfied_set = {g["criterion"] for g in cgm_grades if g["satisfied"]}
        
        retained = baseline_satisfied_set.intersection(cgm_satisfied_set)
        
        if len(baseline_satisfied_set) > 0:
            fidelity = len(retained) / len(baseline_satisfied_set)
        else:
            fidelity = 1.0  # If baseline recalled nothing, and CGM recalled nothing, fidelity is 100%
            
        return {
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "baseline_score": baseline_score,
            "cgm_score": cgm_score,
            "factual_fidelity": fidelity,
            "baseline_grades": baseline_grades,
            "cgm_grades": cgm_grades,
            "total_criteria": total_criteria,
            "baseline_recalled_count": len(baseline_satisfied_set),
            "cgm_recalled_count": len(cgm_satisfied_set),
            "retained_count": len(retained)
        }
