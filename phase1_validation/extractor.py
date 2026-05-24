import json
from typing import Dict, Any, List
from phase1_validation.config import GeminiClient

EXTRACTION_SYSTEM_INSTRUCTION = """You are a Conversational Graph Memory (CGM) Extractor.
Your task is to analyze a multi-turn conversation between a User and an AI Assistant, and extract its core semantic meaning.

You must output a JSON object containing:
1. "summary": A concise 2-4 sentence abstractive summary of the conversation history (key decisions, current project state, user preferences, and goals).
2. "entities": A list of entities mentioned in the conversation. Each entity should have:
   - "name": The unique name of the entity (normalized).
   - "type": One of [Technology, Goal, Preference, Requirement, CodeContext, Concept, Other].
   - "description": A short explanation of the entity's role in the conversation.
3. "triples": A list of semantic relation triples that represent facts and preferences established in the conversation.
   Each triple must be a list/tuple of 3 strings: [subject, predicate, object].
   - The predicate should be a verb or relation like: "prefers", "uses", "requests", "stated_goal_is", "has_error", "seeks_solution_for", "implements", "configured_with".
   - Keep the terms simple, clear, and consistent.
   - Example: ["User", "prefers", "Python"], ["Project", "uses", "FastAPI"], ["Database", "has_error", "ConnectionTimeout"].

Ensure that your output is strictly valid JSON matching this structure. Do not include markdown block formatting (like ```json ... ```) outside the JSON text."""

class GraphExtractor:
    """
    Extracts the Hybrid Memory Object (HMO) components (triples, summary, entities)
    from raw conversation text using Gemini's JSON mode.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def extract(self, conversation_text: str, max_parse_retries: int = 3) -> Dict[str, Any]:
        """
        Processes conversation text and returns a dictionary with extracted triples, entities, and summary.
        Implements a retry loop in case the model returns slightly malformed JSON.
        """
        prompt = f"Please extract the semantic graph and summary from the following conversation:\n\n{conversation_text}"
        
        for attempt in range(max_parse_retries):
            try:
                raw_response = self.client.generate(
                    prompt=prompt,
                    system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
                    json_mode=True
                )
                # Parse response as JSON
                data = json.loads(raw_response.strip())
                return data
            except json.JSONDecodeError as jde:
                print(f"      [WARNING] JSON decode failed (Attempt {attempt+1}/{max_parse_retries}): {jde}")
                if attempt == max_parse_retries - 1:
                    print(f"      [ERROR] JSON parsing failed after {max_parse_retries} attempts.")
                    break
            except Exception as e:
                print(f"      [ERROR] Extraction attempt {attempt+1} failed with error: {e}")
                if attempt == max_parse_retries - 1:
                    break
                    
        # Return empty structure as fallback if all retries fail
        return {
            "summary": "Failed to extract summary.",
            "entities": [],
            "triples": []
        }

    @staticmethod
    def format_memory_as_prompt(memory_data: Dict[str, Any]) -> str:
        """
        Formats the extracted semantic memory into a clean, text-based injection prompt
        that is appended to the LLM's context window.
        """
        summary = memory_data.get("summary", "")
        triples = memory_data.get("triples", [])
        entities = memory_data.get("entities", [])
        
        formatted = "=== CONVERSATIONAL GRAPH MEMORY (DISTILLED HISTORY) ===\n"
        formatted += f"Summary of Past Turns:\n{summary}\n\n"
        
        formatted += "Key Entities Established:\n"
        for entity in entities:
            name = entity.get("name", "")
            etype = entity.get("type", "")
            desc = entity.get("description", "")
            formatted += f"- {name} ({etype}): {desc}\n"
            
        formatted += "\nSemantic Relationships (Facts & Preferences):\n"
        for triple in triples:
            if len(triple) == 3:
                subj, pred, obj = triple
                formatted += f"- <{subj}> ---({pred})---> <{obj}>\n"
                
        formatted += "========================================================\n"
        return formatted
