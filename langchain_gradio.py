
import enum
from typing import Optional, List, Tuple, Dict, Any, Union, Callable

import asyncio
import queue
from concurrent.futures.thread import ThreadPoolExecutor

from langchain.llms.base import BaseLLM
from langchain.chains.base import Chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from langchain.llms.openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI

from dotenv import load_dotenv
import gradio as gr

load_dotenv()


class LLMOutputMode(str, enum.Enum):
    DIRECT = "direct"
    STREAM = "stream"


def get_llm() -> BaseLLM:
    return ChatOpenAI(streaming=True)


def load_chain_module_from_path(chain_path: str, perform_reload: bool = False):
    from importlib import import_module, reload
    module_name, chain_var = chain_path.rsplit(':', 1)
    module = import_module(module_name, '.')
    if perform_reload:
        module = reload(module)
    return module, chain_var


def load_chain(chain_path: str, llm: BaseLLM, perform_reload: bool = False):
    if chain_path:
        chain_module, chain_var = load_chain_module_from_path(chain_path, perform_reload)
        chain_factory: Callable[[BaseLLM], Chain] = getattr(chain_module, chain_var)
        return chain_factory(llm=llm)


def reload_chain(chain_path: str, llm: BaseLLM):
    return load_chain(chain_path, llm, perform_reload=True)


async def handle_upload(file):
    from doc_ingest import upload_file
    return await asyncio.get_event_loop().run_in_executor(None, upload_file, file)


class QueueCallbackHandler(BaseCallbackHandler):
    _queue: Union[queue.Queue, asyncio.Queue]

    def __init__(self, queue: Union[queue.Queue, asyncio.Queue]):
        self._queue = queue

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._queue.put_nowait(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        pass


def run_chain_sync(q: asyncio.Queue, chain: Chain, inputs: Union[str, Dict[str, str]]):
    stream_callback = QueueCallbackHandler(queue=q)
    result = chain(inputs, callbacks=[stream_callback])
    return '\n\n'.join(('%s:\n' % key.upper() if i > 0 else '') + result[key] for i, key in enumerate(chain.output_keys))

class ChatWrapper:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._exec = ThreadPoolExecutor(2)

    async def prepare(self, user_message: str, history, chain_path: str, loaded_chain: Optional[Chain]):
        if loaded_chain is None:
            if not chain_path:
                raise ValueError("Provide path to the chain object")
            raise RuntimeError("Chain is still loading. Please try again in a bit.")

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
        chat_idx = -1 # The latest chat message. We will be writing output to this one
        chat_output: Tuple[str, str] = history[chat_idx]

        if len(user_message) == 0:
            yield history
            return

        # Hold lock till done
        async with self._lock:
            llm_args = [chain, user_message]

            output_queue = asyncio.Queue()
            output_task = asyncio.get_running_loop().run_in_executor(
                self._exec, run_chain_sync, output_queue, *llm_args)

            chat_output[1] = "Generating..."
            yield history

            if mode == LLMOutputMode.DIRECT:
                chat_output[1] = await output_task
                yield history
            elif mode == LLMOutputMode.STREAM:
                fail_count = 0
                chat_output[1] = ""
                while not output_task.done() and fail_count < 3:
                    try:
                        token_task = asyncio.ensure_future(output_queue.get())

                        done, pending = await asyncio.wait([token_task, output_task], return_when=asyncio.FIRST_COMPLETED, timeout=10)

                        if token_task.done():
                            chat_output[1] += await token_task
                            output_queue.task_done()
                            yield history
                        elif token_task in pending:
                            token_task.cancel()
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        fail_count += 1
                    else:
                        fail_count = 0
                # chat_output[1] = await output_task
                await output_task
                yield history
            else:
                yield history
                return

with gr.Blocks().queue(20) as demo:
    with gr.Row(equal_height=False):
        gr.Markdown("# PrivateGPT Demo")

        with gr.Row():
            tb_chain_path = gr.Textbox(
                placeholder="Path to the chain module",
                show_label=True,
                lines=1,
                type="text",
                scale=3
            )
            reload_chain_btn = gr.Button("Reload chain", size='sm')
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

    loaded_llm = gr.State(get_llm)
    loaded_chain = gr.State()
    prepared_message = gr.State('')

    clr_msg_box = lambda: ''

    submit.click(chat_engine.prepare, inputs=[txt_message, chatbot, tb_chain_path, loaded_chain], outputs=[chatbot, prepared_message]) \
        .success(clr_msg_box, outputs=txt_message) \
        .success(chat_engine.generate, inputs=[prepared_message, chatbot, rd_output_mode, loaded_chain], outputs=chatbot)

    txt_message.submit(chat_engine.prepare, inputs=[txt_message, chatbot, tb_chain_path, loaded_chain], outputs=[chatbot, prepared_message]) \
        .success(clr_msg_box, outputs=txt_message) \
        .success(chat_engine.generate, inputs=[prepared_message, chatbot, rd_output_mode, loaded_chain], outputs=chatbot)

    tb_chain_path.change(load_chain, inputs=[tb_chain_path, loaded_llm], outputs=loaded_chain)
    reload_chain_btn.click(reload_chain, inputs=[tb_chain_path, loaded_llm], outputs=loaded_chain)
    file_upload_box.upload(handle_upload, inputs=[file_upload_box])

if __name__ == "__main__":
    demo.launch(debug=True)
