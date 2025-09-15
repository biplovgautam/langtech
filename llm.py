import requests
import os
from typing import Optional, List
from langchain.llms.base import LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqLLM(LLM):
    """Custom LangChain wrapper for Groq's chat API with API key embedded"""

    # Automatically load from environment
    api_key: str = os.getenv("GROQ_API_KEY")

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Groq API with error handling"""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "openai/gpt-oss-120b",
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"❌ Network error calling Groq API: {e}"
        except KeyError as e:
            return f"❌ Unexpected response format from Groq API: {e}"
        except Exception as e:
            return f"❌ Error calling Groq API: {e}"
