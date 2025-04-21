# app.py
import gradio as gr
import uuid # To generate unique session IDs
import logging

# Import the initialized chain from the logic module
from src.chatbot_logic import conversational_rag_chain, get_session_history

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if chatbot initialization failed
if conversational_rag_chain is None:
    logger.error("Chatbot chain failed to initialize. Exiting.")
    # Optionally, display an error message in Gradio instead of exiting
    # raise RuntimeError("Chatbot initialization failed. Check logs.")
    exit(1)

# Function to handle user interactions with streaming
def respond(message, chat_history, session_id):
    """Handles user message input, invokes the RAG chain, and streams the response."""
    logger.info(f"Session [{session_id}]: Received message: {message}")
    if not message:
        logger.warning(f"Session [{session_id}]: Empty message received.")
        yield chat_history # Return history unchanged if message is empty
        return

    # Append user message immediately
    chat_history.append((message, ""))

    response_stream = conversational_rag_chain.stream(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )

    full_response = ""
    try:
        for chunk in response_stream:
            # Extract the actual response chunk from the stream output
            response_part = chunk.get("answer", "")
            if response_part:
                full_response += response_part
                # Update the last entry in chat_history with the streamed response
                chat_history[-1] = (message, full_response)
                yield chat_history # Yield the updated history to Gradio
            # Handle other potential keys in the chunk if needed (e.g., context)
            # retrieved_context = chunk.get("context", [])
            # logger.debug(f"Session [{session_id}]: Retrieved context: {retrieved_context}")

    except Exception as e:
        logger.error(f"Session [{session_id}]: Error during streaming: {e}", exc_info=True)
        chat_history[-1] = (message, f"An error occurred: {e}")
        yield chat_history # Show error to user

    logger.info(f"Session [{session_id}]: Full response generated.")


# Function to handle the Retry button
def retry_fn(chat_history, session_id):
    """Retries the last user message."""
    if not chat_history or not chat_history[-1][0]:
        logger.warning(f"Session [{session_id}]: No message to retry.")
        return chat_history # Return history unchanged if no message to retry

    logger.info(f"Session [{session_id}]: Retrying last message.")
    last_user_message = chat_history[-1][0]
    # Remove the potentially incomplete previous attempt
    chat_history.pop()
    # Stream the response again
    # Use 'yield from' to delegate yielding to the respond generator
    yield from respond(last_user_message, chat_history, session_id)


# Function to handle the Undo button
def undo_fn(chat_history, session_id):
    """Removes the last user message and the chatbot's response."""
    if chat_history:
        logger.info(f"Session [{session_id}]: Undoing last interaction.")
        chat_history.pop()
        # Also clear the corresponding history in the backend store (optional but good practice)
        # Note: This simplistic undo doesn't perfectly revert the LangChain history object state
        # for complex stateful chains, but is okay for basic Q&A history.
        session_history = get_session_history(session_id)
        if len(session_history.messages) >= 2:
             session_history.messages.pop() # Remove assistant message
             session_history.messages.pop() # Remove human message
    else:
         logger.warning(f"Session [{session_id}]: No interaction to undo.")
    return chat_history

# Function to handle the Clear button
def clear_fn(session_id):
    """Clears the chat history for the session."""
    logger.info(f"Session [{session_id}]: Clearing chat history.")
    session_history = get_session_history(session_id)
    session_history.clear()
    return [] # Return empty list to clear Gradio Chatbot UI

# Create Gradio interface using Blocks for more control
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Interactive CV Chatbot (RAG Demo)
        Ask questions about the loaded CV document. The chatbot uses Retrieval-Augmented Generation.
        """
    )

    # Hidden state to store the unique session ID for each user connection
    session_id_state = gr.State(lambda: str(uuid.uuid4()))

    chatbot_display = gr.Chatbot(label="Chat History", height=500)
    message_input = gr.Textbox(
        label="Type your query here:",
        placeholder="e.g., What was the last project mentioned?",
        lines=2
    )
    submit_button = gr.Button("Send", variant="primary")

    with gr.Row():
        retry_button = gr.Button("🔄 Retry")
        undo_button = gr.Button("↩️ Undo")
        clear_button = gr.Button("🗑️ Clear")

    # Link actions to functions
    submit_button.click(
        fn=respond,
        inputs=[message_input, chatbot_display, session_id_state],
        outputs=[chatbot_display],
        queue=True # Enable queuing for handling multiple users
    ).then(lambda: gr.update(value=""), None, [message_input], queue=False) # Clear input after send

    message_input.submit(
         fn=respond,
        inputs=[message_input, chatbot_display, session_id_state],
        outputs=[chatbot_display],
        queue=True
    ).then(lambda: gr.update(value=""), None, [message_input], queue=False) # Clear input after send


    retry_button.click(
        fn=retry_fn,
        inputs=[chatbot_display, session_id_state],
        outputs=[chatbot_display],
        queue=True
    )

    undo_button.click(
        fn=undo_fn,
        inputs=[chatbot_display, session_id_state],
        outputs=[chatbot_display],
        queue=False # Undo is usually quick
    )

    clear_button.click(
        fn=clear_fn,
        inputs=[session_id_state],
        outputs=[chatbot_display],
        queue=False # Clear is usually quick
    )

# Launch the Gradio interface
if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.queue() # Enable queuing for smoother streaming and multi-user experience
    demo.launch(inbrowser=True)
    logger.info("Gradio interface stopped.")