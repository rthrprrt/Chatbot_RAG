# src/config.py
import os
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class AppConfig:
    """Loads application configuration from environment variables."""

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo-0125")

    # RAG Document
    PDF_PATH = os.getenv("PDF_PATH", "data/example.pdf") # Default path

    # Prompts
    QA_SYSTEM_PROMPT = os.getenv(
        "QA_SYSTEM_PROMPT",
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"
    )
    CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    # Text Splitter
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # Vector Store (Optional persistence)
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH") # If None, use in-memory

    @staticmethod
    def validate():
        """Basic validation for critical configurations."""
        if not AppConfig.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            # Attempt to get it via input as a fallback (optional)
            try:
                 AppConfig.OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ")
                 if not AppConfig.OPENAI_API_KEY:
                     raise ValueError("OpenAI API Key is required.")
            except Exception:
                 raise ValueError("OpenAI API Key is required but was not provided.")

        if not os.path.exists(AppConfig.PDF_PATH):
            raise FileNotFoundError(
                f"PDF document not found at path specified in PDF_PATH: {AppConfig.PDF_PATH}"
            )
        print("Configuration loaded and validated.")

# Load and validate config on import
try:
    CONFIG = AppConfig()
    CONFIG.validate()
except (ValueError, FileNotFoundError) as e:
    print(f"Configuration Error: {e}")
    print("Please ensure your .env file is set up correctly and the PDF file exists.")
    exit(1) # Exit if critical config is missing or invalid