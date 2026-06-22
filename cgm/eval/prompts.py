"""Shared prompt templates for training, inference, and benchmark evaluation."""

SYSTEM_PREFIX = (
    "System: You are a helpful assistant with access to conversation memory. "
    "Answer the user's question concisely and directly based on the provided memory. "
    "Give short answers. "
    "If the answer is not in the memory, say 'I do not know.' "
    "Note that 'User' in the memory refers to the current user.\n"
)

LOCOMO_SYSTEM_PREFIX = (
    "System: You are a helpful assistant. Answer the question based on the "
    "conversation history. Give a short, direct answer (1-2 sentences max). "
    "If the information is not available, say 'No information available.'\n"
)

# Optimized for CGM mode on LoCoMo benchmark: concise factual answers
LOCOMO_CGM_PREFIX = (
    "System: You are an assistant with access to conversation memory. "
    "Answer with a short, direct factual response. "
    "Do not repeat the question. Do not explain. "
    "If you don't know, say 'I do not know.'\n"
)


def build_user_prompt(question: str, system_prefix: str = SYSTEM_PREFIX) -> str:
    """Format a question for CGM pipeline generate (matches inference path)."""
    return f"{system_prefix}User: {question.strip()}\nAssistant:"


def build_locomo_prompt(question: str, context: str = "") -> str:
    """Format a LoCoMo QA prompt with optional retrieved/stuffed context."""
    if context.strip():
        return f"{LOCOMO_SYSTEM_PREFIX}{context.strip()}\n\nUser: {question.strip()}\nAssistant:"
    return f"{LOCOMO_SYSTEM_PREFIX}User: {question.strip()}\nAssistant:"


def wrap_training_prompt(prompt_text: str, system_prefix: str = SYSTEM_PREFIX) -> str:
    """Prepend system prefix to training prompts if not already present."""
    if prompt_text.startswith("System:"):
        return prompt_text
    return f"{system_prefix}{prompt_text}"
