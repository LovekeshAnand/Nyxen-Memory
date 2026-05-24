import os
import sys
import time
import requests
from dotenv import load_dotenv

# Try to find the .env file in the parent directory as well
# Since the script runs from d:\Nyxen-Memory\phase1_validation\, the .env is in d:\Nyxen-Memory\
possible_dotenv_paths = [
    os.path.join(os.getcwd(), '.env'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
]

dotenv_loaded = False
for path in possible_dotenv_paths:
    if os.path.exists(path):
        load_dotenv(path)
        dotenv_loaded = True
        break

if not dotenv_loaded:
    load_dotenv() # Default fallback

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiClient:
    """
    A simple client for interacting with the Google Gemini API.
    Does not require any heavy ML libraries, only the standard 'requests' library.
    """
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            print("[ERROR] GEMINI_API_KEY was not found.")
            print("Please ensure you have a .env file containing:")
            print("GEMINI_API_KEY=your_actual_api_key_here")
            print("You can get a free API key from Google AI Studio: https://aistudio.google.com/")
            sys.exit(1)
            
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Throttling to respect the 5 RPM rate limit of the free tier
        self.last_request_time = 0.0
        self.min_interval = 12.5  # 12.5 seconds minimum between requests

    def generate(self, prompt: str, system_instruction: str = None, json_mode: bool = False, max_retries: int = 5) -> str:
        """
        Sends a text prompt to the Gemini model and returns the text response.
        Handles API errors, rate limits (HTTP 429), and implements exponential backoff.
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction}
                ]
            }
            
        if json_mode:
            payload["generationConfig"] = {
                "responseMimeType": "application/json"
            }
            
        backoff_factor = 2.0
        for attempt in range(max_retries):
            # Apply proactive throttling to stay under the 5 RPM free tier limit
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                print(f"      [Rate Limiter] Waiting {sleep_time:.1f}s to respect the 5 RPM API limit...")
                time.sleep(sleep_time)
                
            try:
                # Update last request timestamp before we execute the request
                self.last_request_time = time.time()
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                
                # Check for rate limit (HTTP 429)
                if response.status_code == 429:
                    retry_delay = 10.0  # default fallback
                    try:
                        err_data = response.json()
                        details = err_data.get("error", {}).get("details", [])
                        for detail in details:
                            if "retryDelay" in detail:
                                delay_str = detail["retryDelay"]
                                # E.g. "52s" -> 52.0
                                if delay_str.endswith("s"):
                                    retry_delay = float(delay_str[:-1]) + 1.5
                                break
                    except Exception:
                        pass
                    
                    print(f"      [WARNING] Gemini API rate limit hit (429). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                    
                if response.status_code != 200:
                    raise Exception(f"Gemini API error ({response.status_code}): {response.text}")
                    
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
                
            except Exception as e:
                # If we've run out of retries, raise the error
                if attempt == max_retries - 1:
                    print(f"      [ERROR] Gemini request failed after {max_retries} attempts: {e}")
                    raise e
                
                # Otherwise, sleep with exponential backoff and try again
                sleep_time = (backoff_factor ** attempt) + 2.0
                print(f"      [WARNING] Request failed: {e}. Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)

