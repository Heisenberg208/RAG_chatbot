import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", )
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER","ollama")
    LLM_MODEL=os.getenv("LLM_MODEL","lama3.2:3b")
    WEB_URLS = os.getenv("WEB_URLS", "")
    WEB_URLS = [url.strip() for url in WEB_URLS.split(",") if url.strip()]
    # Paths
    DATA_PATH = Path("data")
    VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", "./vectorstore"))
    
    # Text Processing
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    TOP_K_RESULTS=5
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def setup_logging(cls):
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )