import time
import gradio as gr
from chatbot import conversational_rag_chain

# Gradio function to handle chatbot interaction with streaming
def chatbot_response(session_id, user_input):
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    # Simulate streaming by chunking the response
    chunk_size = 50  # Adjust chunk size as needed
    for i in range(0, len(response), chunk_size):
        yield response[i:i+chunk_size]

# Function to handle user interactions
def respond(message, chat_history):
    session_id = "default_session"
    response_stream = chatbot_response(session_id, message)
    for response in response_stream:
        if chat_history and isinstance(chat_history[-1], tuple) and len(chat_history[-1]) == 2:
            chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + response)
        else:
            chat_history.append((message, response))
        yield "", chat_history

def retry_fn(chat_history):
    if chat_history:
        session_id = "default_session"
        message = chat_history[-1][0]
        response_stream = chatbot_response(session_id, message)
        for response in response_stream:
            if chat_history and isinstance(chat_history[-1], tuple) and len(chat_history[-1]) == 2:
                chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + response)
            else:
                chat_history.append((message, response))
            yield "", chat_history

def undo_fn(chat_history):
    if chat_history:
        chat_history.pop()
    return "", chat_history

# Create Gradio interface
with gr.Blocks(theme='gradio/soft') as demo:
    gr.Markdown("# Arthur Perrot Interactive CV Chatbot\nChatbot using RAG")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your query here:", placeholder="Bonjour, peux tu me décrire ta dernière expérience professionnelle ?")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    with gr.Row():
        retry_button = gr.Button("Retry", variant="primary")
        undo_button = gr.Button("Undo", variant="primary")
        clear = gr.ClearButton([msg, chatbot], variant="secondary")

    retry_button.click(
        fn=retry_fn,
        inputs=chatbot,
        outputs=[msg, chatbot],
    )

    undo_button.click(
        fn=undo_fn,
        inputs=chatbot,
        outputs=[msg, chatbot],
    )

# Launch the Gradio interface
demo.launch(inbrowser=True, inline=False)
