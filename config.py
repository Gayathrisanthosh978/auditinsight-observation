import os
from dotenv import load_dotenv

load_dotenv()

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:70b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

# NVIDIA LLM Model configuration
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4")
NVIDIA_LLM_BASE_URL = os.getenv("NVIDIA_LLM_BASE_URL", "http://0.0.0.0:8000/v1")
