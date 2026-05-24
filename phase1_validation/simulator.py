from typing import Dict, Any, List
from phase1_validation.config import GeminiClient

SCENARIOS = [
    {
        "id": 1,
        "name": "FastAPI & PostgreSQL Backend Setup",
        "description": "User sets up database, typing, port, framework, and linting preferences over 5 turns.",
        "history": [
            {"role": "user", "text": "I want to build a backend API using FastAPI. It should connect to PostgreSQL and run on port 8080. Can you set up the boilerplate?"},
            {"role": "assistant", "text": "Sure! I can write a FastAPI boilerplate that connects to PostgreSQL and runs on port 8080. Let me know what database libraries and other tools you prefer."},
            {"role": "user", "text": "Actually, let's use 'asyncpg' for database connectivity instead of SQLAlchemy, and I prefer 'typing_extensions' for type hints. Also, can we use Pydantic v2?"},
            {"role": "assistant", "text": "Got it. I will use asyncpg directly for asynchronous PostgreSQL connections, typing_extensions for compatibility, and Pydantic v2 for data validation schemas. What schema should we define first?"},
            {"role": "user", "text": "For the database table, let's name it 'user_profiles' and it should contain columns: id (UUID), username (VARCHAR 50, unique), email (VARCHAR 255), and created_at (TIMESTAMP)."},
            {"role": "assistant", "text": "Understood. The 'user_profiles' table will have: id (UUID), username (VARCHAR 50, unique), email (VARCHAR 255), and created_at (TIMESTAMP). Ready when you are."},
            {"role": "user", "text": "Awesome. I also prefer strict linting. Let's make sure we use Ruff for formatting and linting, and my preferred line length is 100 characters."},
            {"role": "assistant", "text": "Excellent. I've noted Ruff as the formatter/linter with a max line length of 100 characters. Is there anything else we need to save before writing code?"},
            {"role": "user", "text": "Perfect. Let's save all this configuration."}
        ],
        "test_query": "Now, let's write the Pydantic schema for database insertion. Write the code according to all the preferences, database column names, port configurations, and tools we agreed on in the earlier turns. Do not ask me for them.",
        "criteria": [
            "Uses FastAPI as the web framework",
            "Uses Pydantic v2 syntax (e.g. BaseModel, Field or EmailStr)",
            "Configures or mentions database port/connection for PostgreSQL",
            "Uses or mentions asyncpg for database connection",
            "Defines the table or schema named 'user_profiles'",
            "Includes columns: id (UUID), username (VARCHAR 50, unique), email (VARCHAR 255), and created_at (TIMESTAMP)"
        ]
    },
    {
        "id": 2,
        "name": "Docker Container Connection Debugging",
        "description": "User debugs a connection issue between two Docker containers and sets up environmental variables.",
        "history": [
            {"role": "user", "text": "I am hosting a service on Docker. The service name is 'auth-api'. It is running on a bridge network named 'app-net' at port 5000."},
            {"role": "assistant", "text": "Noted. You have a service container named 'auth-api' running on the 'app-net' bridge network, listening on port 5000."},
            {"role": "user", "text": "I am trying to connect to it from another container 'frontend-web' on the same network 'app-net', but I keep getting a ConnectionRefused error. The frontend container is using the URL 'http://localhost:5000/validate'."},
            {"role": "assistant", "text": "Ah! Inside a Docker container, 'localhost' refers to the container itself (frontend-web), not the host or other containers. To connect to 'auth-api', you should use the container name as the hostname: 'http://auth-api:5000/validate'."},
            {"role": "user", "text": "Right, the service name is 'auth-api' so the URL should be 'http://auth-api:5000/validate'. Let's verify that."},
            {"role": "assistant", "text": "Correct. Using the container name resolves to the correct internal container IP address inside the 'app-net' network. What other configurations do we have?"},
            {"role": "user", "text": "We also set an environment variable SECURE_MODE=true on the 'auth-api' container."},
            {"role": "assistant", "text": "Got it. SECURE_MODE=true is set on the 'auth-api' container. The frontend can now access the endpoint at http://auth-api:5000/validate."}
        ],
        "test_query": "Write out the final environment configuration or compose setup for both docker containers and the updated connection URL we configured, keeping in mind the secure mode setting we set on auth-api. Summarize the exact Docker network details and hostnames we agreed to use.",
        "criteria": [
            "Mentions or sets up the Docker network named 'app-net'",
            "Uses the connection URL 'http://auth-api:5000/validate'",
            "Sets the container/service name for the auth API as 'auth-api'",
            "Sets the environment variable SECURE_MODE=true on the auth-api container",
            "Identifies the container name 'frontend-web' as the frontend container"
        ]
    }
]

class DialogueSimulator:
    """
    Simulates dialogue runs under two conditions:
    1. Baseline: Uses the full conversation history.
    2. CGM: Discards history, injecting only the extracted semantic graph memory prompt.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def format_history_as_text(self, history: List[Dict[str, str]]) -> str:
        """
        Formats a structured dialogue history list into a single readable string.
        """
        formatted = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            formatted += f"{role}: {turn['text']}\n\n"
        return formatted

    def run_baseline(self, scenario: Dict[str, Any]) -> str:
        """
        Runs the test query with the entire conversation history in context.
        """
        history_text = self.format_history_as_text(scenario["history"])
        prompt = f"{history_text}User: {scenario['test_query']}\nAssistant:"
        
        system_instruction = "You are a helpful AI software development and systems assistant. Answer the user's latest query using the history provided above."
        
        return self.client.generate(prompt=prompt, system_instruction=system_instruction)

    def run_cgm(self, scenario: Dict[str, Any], memory_prompt: str) -> str:
        """
        Runs the test query with ONLY the memory prompt (injected graph summary + triples) and the new query.
        Discards the raw message history.
        """
        # We prepend the memory prompt and then append the final query
        prompt = f"{memory_prompt}\n\nUser Query: {scenario['test_query']}\nAssistant:"
        
        system_instruction = "You are a helpful AI software development and systems assistant. You are provided with a 'Conversational Graph Memory' containing distilled facts from past turns. Answer the user's query according to these facts."
        
        return self.client.generate(prompt=prompt, system_instruction=system_instruction)
