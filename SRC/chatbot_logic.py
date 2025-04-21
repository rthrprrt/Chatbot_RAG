# src/chatbot_logic.py
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CONFIG # Import from sibling module

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Store for Session History (In-Memory - Limitation) ---
# Warning: This store is volatile and will be lost on application restart.
# For production, use a persistent store (e.g., Redis, database).
_session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or creates a chat message history for a given session ID."""
    if session_id not in _session_store:
        logger.info(f"Creating new chat history for session_id: {session_id}")
        _session_store[session_id] = ChatMessageHistory()
    else:
        logger.debug(f"Using existing chat history for session_id: {session_id}")
    return _session_store[session_id]

def create_rag_chain(llm, retriever):
    """Creates the conversational RAG chain with history."""
    logger.info("Creating RAG chain...")

    # 1. Contextualize Question Chain
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONFIG.CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    logger.info("History-aware retriever created.")

    # 2. Question Answering Chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONFIG.QA_SYSTEM_PROMPT), # Loaded from config
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    logger.info("Question-answering chain created.")

    # 3. Combine into Retrieval Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    logger.info("Base RAG chain created.")

    # 4. Add Message History Wrapper
    conversational_rag_chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    logger.info("Conversational RAG chain with history created.")
    return conversational_rag_chain_with_history

def initialize_chatbot():
    """Initializes all components needed for the chatbot."""
    logger.info("Initializing chatbot components...")

    # Initialize LLM
    llm = ChatOpenAI(
        model=CONFIG.OPENAI_MODEL_NAME,
        openai_api_key=CONFIG.OPENAI_API_KEY,
        streaming=True # Enable streaming from the LLM
    )
    logger.info(f"LLM initialized: {CONFIG.OPENAI_MODEL_NAME}")

    # Load and Split Document
    try:
        loader = PyPDFLoader(CONFIG.PDF_PATH)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} document pages from {CONFIG.PDF_PATH}")
    except Exception as e:
        logger.error(f"Error loading PDF from {CONFIG.PDF_PATH}: {e}")
        raise  # Re-raise after logging

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.CHUNK_SIZE,
        chunk_overlap=CONFIG.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split document into {len(splits)} chunks.")

    # Create Vector Store and Retriever
    logger.info("Creating vector store...")
    embeddings = OpenAIEmbeddings(openai_api_key=CONFIG.OPENAI_API_KEY)

    # Optional persistence
    persist_directory = CONFIG.VECTORSTORE_PATH if CONFIG.VECTORSTORE_PATH else None
    if persist_directory and os.path.exists(persist_directory):
         logger.info(f"Loading existing vector store from {persist_directory}")
         vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
         logger.info(f"Creating new vector store {'at ' + persist_directory if persist_directory else '(in-memory)'}")
         vectorstore = Chroma.from_documents(
             documents=splits,
             embedding=embeddings,
             persist_directory=persist_directory # Will save if path is provided
         )
    retriever = vectorstore.as_retriever()
    logger.info("Vector store and retriever created.")

    # Create the full conversational chain
    chain = create_rag_chain(llm, retriever)
    logger.info("Chatbot initialization complete.")
    return chain

# --- Initialize the chain globally on module load ---
# This assumes the config is valid at this point
try:
    conversational_rag_chain = initialize_chatbot()
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    # Handle initialization failure appropriately, maybe exit or provide a dummy chain
    conversational_rag_chain = None # Or raise an error to prevent app start