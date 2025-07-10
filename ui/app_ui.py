"""
Gradio UI for the RAG Support Assistant
"""

import gradio as gr
import requests
import json
from typing import Generator

# Configuration for the FastAPI backend
BASE_URL = "http://localhost:8000"
GENERATE_STREAM_URL = f"{BASE_URL}/generate_stream"

def get_llm_providers():
    # In a real app, this could be fetched from the backend
    return ["auto", "local", "gemini"]

def rag_assistant(query: str, provider: str) -> Generator[str, None, None]:
    """
    Function to be called by the Gradio interface.
    It streams the response from the FastAPI backend.
    """
    if not query:
        yield "Please enter a query."
        return

    payload = {
        "query": query,
        "provider": provider,
        "stream": True
    }
    
    full_response = ""
    try:
        with requests.post(GENERATE_STREAM_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    # Handle SSE format if present
                    if decoded_chunk.startswith("data: "):
                        decoded_chunk = decoded_chunk[6:]
                    
                    full_response += decoded_chunk
                    yield full_response

    except requests.exceptions.RequestException as e:
        yield f"Error connecting to the backend: {e}"
    except Exception as e:
        yield f"An error occurred: {e}"

def build_ui():
    """Builds the Gradio UI"""
    with gr.Blocks(theme="soft", title="RAG Support Assistant") as demo:
        gr.Markdown("# 📘 RAG Support Assistant")
        gr.Markdown("Ask a question to the customer support assistant.")

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Why is the API returning a 403 error?",
                    lines=3
                )
                provider_dropdown = gr.Dropdown(
                    label="Select LLM Provider",
                    choices=get_llm_providers(),
                    value="auto"
                )
                submit_button = gr.Button("Ask Assistant", variant="primary")
            
            with gr.Column(scale=6):
                response_output = gr.Markdown(label="Assistant's Response")

        submit_button.click(
            fn=rag_assistant,
            inputs=[query_input, provider_dropdown],
            outputs=[response_output]
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860) 