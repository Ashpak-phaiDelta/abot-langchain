
from langchain.chains.base import Chain
from typing import Optional, Tuple, List

import time
import enum
import gradio as gr
import asyncio
from dotenv import load_dotenv
from concurrent.futures.thread import ThreadPoolExecutor

load_dotenv()


class LLMOutputMode(str, enum.Enum):
    DIRECT = "direct"
    STREAM = "stream"


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


def run_chain_sync(q: asyncio.Queue, chain: Chain, input: str):
    # TODO: Custom callback class to append to queue
    result = chain(input, callbacks=[])
    # for s in msg:
    #     time.sleep(1)
    #     q.put_nowait(s)
    return result[chain.output_keys[0]]

class ChatWrapper:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._exec = ThreadPoolExecutor(2)

    async def prepare(self, user_message: str, history, chain_path: str, loaded_chain: Optional[Chain]):
        if loaded_chain is None:
            if not chain_path:
                raise ValueError("Provide path to the chain object")
            raise TimeoutError("Chain is still loading. Please try again in a bit.")

        if self._lock.locked():
            raise RuntimeError("A chat operation is still in progress. Please wait till it finishes.")

        user_message = user_message.strip()
        if len(user_message) > 0:
            async with self._lock:
                new_message_pair = [user_message, None]
                history.append(new_message_pair)
        return history, user_message

    async def generate(
        self,
        user_message: str,
        history,
        mode: LLMOutputMode,
        chain: Chain
        ):
        chat_idx = -1
        chat_output = history[chat_idx]

        if len(user_message) == 0:
            yield history
            return

        # Hold lock till done
        async with self._lock:
            llm_args = [chain, user_message]

            output_queue = asyncio.Queue()
            output_task = asyncio.get_running_loop().run_in_executor(
                self._exec, run_chain_sync, output_queue, *llm_args)

            if mode == LLMOutputMode.DIRECT:
                chat_output[1] = "Generating..."
                yield history
                chat_output[1] = await output_task
                yield history
            elif mode == LLMOutputMode.STREAM:
                fail_count = 0
                chat_output[1] = ""
                while (not output_task.done()) and fail_count < 3:
                    try:
                        tok = await asyncio.wait_for(output_queue.get(), timeout=2)
                        chat_output[1] += tok
                        yield history
                    except (asyncio.CancelledError, asyncio.QueueEmpty):
                        fail_count += 1
                    else:
                        fail_count = 0
                chat_output[1] = await output_task
                yield history
            else:
                yield history
                return

with gr.Blocks().queue(20) as demo:
    with gr.Row(equal_height=False):
        gr.Markdown("# PrivateGPT Demo")

        with gr.Row():
            reload_chain_btn = gr.Button("Reload chain", size='sm')
            tb_chain_path = gr.Textbox(
                placeholder="Path to the chain module",
                show_label=False,
                lines=1,
                type="text",
                scale=3
            )
        gr.Examples(
            examples=["doc_parse:ask_doc_chain", "genesis.chat_chain:agent_chain"],
            inputs=tb_chain_path,
            examples_per_page=3
        )
        rd_output_mode = gr.Radio(
            [LLMOutputMode.DIRECT, LLMOutputMode.STREAM],
            value=LLMOutputMode.DIRECT,
            label="Output mode",
            interactive=True
        )

    chat_engine = ChatWrapper()
    chatbot = gr.Chatbot()

    with gr.Row():
        txt_message = gr.Textbox(
            label="Chat",
            placeholder="Example: Describe the contents",
            lines=1,
            scale=11
        )
        submit = gr.Button(value="Send", variant="secondary", scale=1)

    with gr.Row():
        file_upload_box = gr.File()

    gr.Examples(
        examples=[
            "Hi!",
            "Whats 2 + 2?",
            "Talk about the documents loaded",
            "List of documents"
        ],
        inputs=txt_message
    )

    gr.HTML("Demo application of a Langchain-based PrivateGPT.")

    loaded_chain = gr.State()
    prepared_message = gr.State('')

    clr_msg_box = lambda: ''

    submit.click(chat_engine.prepare, inputs=[txt_message, chatbot, tb_chain_path, loaded_chain], outputs=[chatbot, prepared_message]) \
        .success(clr_msg_box, outputs=txt_message) \
        .then(chat_engine.generate, inputs=[prepared_message, chatbot, rd_output_mode, loaded_chain], outputs=chatbot)

    txt_message.submit(chat_engine.prepare, inputs=[txt_message, chatbot, tb_chain_path, loaded_chain], outputs=[chatbot, prepared_message]) \
        .success(clr_msg_box, outputs=txt_message) \
        .then(chat_engine.generate, inputs=[prepared_message, chatbot, rd_output_mode, loaded_chain], outputs=chatbot)

    tb_chain_path.change(load_chain, inputs=tb_chain_path, outputs=loaded_chain)
    reload_chain_btn.click(reload_chain, inputs=tb_chain_path, outputs=loaded_chain)
    file_upload_box.upload(handle_upload, inputs=[file_upload_box])

if __name__ == "__main__":
    demo.launch(debug=True)
