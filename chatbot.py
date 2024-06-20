import os
import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY") or getpass.getpass("Enter your OpenAI API key: ")

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=openai_api_key, streaming=True)

# Load and split the PDF document
loader = PyPDFLoader("insert_your_pdf")
docs = loader.load()
print(f"Loaded documents: {docs}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Split documents: {splits}")

# Create the vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Define the contextual question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define the enhanced QA system prompt
qa_system_prompt = """Write_your_prompt_here
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the retrieval-augmented generation chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Define a function to manage chat history
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create a chain with message history management
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to stream and display the response
def stream_response(chain, input_data, session_id):
    responses = []
    print(f"Input data: {input_data}")
    for chunk in chain.stream(input_data, config={"configurable": {"session_id": session_id}}):
        response_part = chunk.get("answer", "")
        responses.append(response_part)
        print(f"Chunk response: {response_part}")
        yield "".join(responses)

# Example usage
session_id = "unique_session_id"
user_input = "unique_user_input"

response = conversational_rag_chain.invoke(
    {"input": user_input},
    config={"configurable": {"session_id": session_id}}
)["answer"]
print(response)

# Stream the response
stream_response(conversational_rag_chain, {"input": user_input}, session_id)
