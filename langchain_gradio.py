
from langchain.chains.base import Chain
from typing import Optional, Tuple

import gradio as gr
from threading import Lock
from dotenv import load_dotenv

load_dotenv()


def load_chain_from_module(chain_path: str):
    from importlib import import_module
    module_name, chain_var = chain_path.rsplit(':', 1)
    module = import_module(module_name, '.')
    return module, chain_var


def load_chain(chain_path: str):
    if chain_path:
        chain_module, chain_var = load_chain_from_module(chain_path)
        chain = getattr(chain_module, chain_var)
        return chain

def reload_chain(chain_path: str):
    if chain_path:
        from importlib import reload
        chain_module, chain_var = load_chain_from_module(chain_path)
        chain_module = reload(chain_module)
        chain = getattr(chain_module, chain_var)
        return chain

def handle_upload(file):
    from importlib import import_module, reload
    ingest = import_module("doc_ingest", '.')
    ingest = reload(ingest)
    ingest.upload_file(file)

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
    def __call__(
        self,
        inp: str,
        history: Optional[Tuple[str, str]],
        chain: Optional[Chain],
        tb_chain_path: Optional[str]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                if not tb_chain_path:
                    history.append((inp, "Provide path to the chain object"))
                    return history, history
                
                history.append((inp, "Chain is still loading. Please try again in a bit."))
                return history, history

            # Run chain and append input.
            output = chain(inp)
            history.append((inp, output[chain.output_keys[0]]))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

with gr.Blocks() as demo:
    with gr.Row(equal_height=False):
        gr.Markdown("### PrivateGPT Demo")

        with gr.Row():
            reload_chain_btn = gr.Button("Reload chain")
            tb_chain_path = gr.Textbox(
                placeholder="Path to the chain module",
                show_label=False,
                lines=1,
                type="text",
                scale=2
            )
        gr.Examples(
            examples=["doc_parse:ask_doc_chain", "genesis.chat_chain:agent_chain"],
            inputs=tb_chain_path,
            examples_per_page=3
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ex: Describe the contents",
            lines=1,
            scale=19
        )
        submit = gr.Button(value="Send", variant="secondary", scale=1)

    with gr.Row():
        rd_output_mode = gr.Radio(
            ["Direct", "Stream"],
            label="Output mode"
        )

        file_upload_box = gr.File()

    gr.Examples(
        examples=[
            "Hi!",
            "Whats 2 + 2?",
            "Talk about the documents loaded",
            "List of documents"
        ],
        inputs=message
    )

    gr.HTML("Demo application of a Langchain-based PrivateGPT.")

    chat_history = gr.State()
    loaded_chain = gr.State()

    submit.click(chat, inputs=[message, chat_history, loaded_chain, tb_chain_path], outputs=[chatbot, chat_history])
    message.submit(chat, inputs=[message, chat_history, loaded_chain, tb_chain_path], outputs=[chatbot, chat_history])
    reload_chain_btn.click(reload_chain, inputs=[tb_chain_path], outputs=[loaded_chain])
    tb_chain_path.change(load_chain, inputs=[tb_chain_path], outputs=[loaded_chain])
    file_upload_box.upload(handle_upload, inputs=[file_upload_box])

if __name__ == "__main__":
    demo.launch(debug=True)
