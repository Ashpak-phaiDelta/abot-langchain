
import enum
from typing import Optional, List, Tuple, Dict, Any, Union, Callable

import asyncio
import queue
from concurrent.futures.thread import ThreadPoolExecutor
from tempfile import _TemporaryFileWrapper

from langchain.llms.base import BaseLLM
from langchain.chains.base import Chain
from langchain.vectorstores.base import VectorStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from langchain.llms.openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.vectorstores import Chroma, PGVector

from vectorstores.doc_chroma import chromadb
from vectorstores.genesis_pg import genesisdb
from sqlalchemy.orm import Session

from dotenv import load_dotenv
import gradio as gr
import pandas as pd

load_dotenv()


class LLMOutputMode(str, enum.Enum):
    DIRECT = "direct"
    STREAM = "stream"


TARGET_SOURCE_CHUNKS = 12

ALL_VECTORSTORES: Dict[str, Optional[VectorStore]] = dict(
    none = None,
    chromadb = chromadb,
    genesisdb = genesisdb,
)

CHAIN_EXAMPLES = [
    ["vanilla_llm:simple"],
    ["doc_parse:ask_doc_chain"],
    ["doc_parse:ask_genesis_chain"],
    ["doc_parse:vectorstore_agent"],
    ["genesis.chat_chain:agent_chain"]
]

# vectorstore collection utils

def vs_doc_list(vs: str, *needed_info) -> List:
    vstore = ALL_VECTORSTORES.get(vs)
    all_docs = dict()
    response = []
    if vstore is None:
        return response

    if isinstance(vstore, Chroma):
        result = vstore._collection.get()
        for doc_id, doc_data, metadata in zip(result['ids'], result['documents'], result['metadatas']):
            if metadata.get('source'):
                if metadata['source'] not in all_docs:
                    all_docs[metadata['source']] = ([], [])
                all_docs[metadata['source']][0].append(doc_id)
                all_docs[metadata['source']][1].append(doc_data)
    elif isinstance(vstore, PGVector):
        with Session(vstore._conn) as session:
            collection_data = vstore.get_collection(session)
            if collection_data:
                for embedding in collection_data.embeddings:
                    doc_id = embedding.uuid
                    doc_data = embedding.document
                    metadata = embedding.cmetadata
                    if metadata.get('source'):
                        if metadata['source'] not in all_docs:
                            all_docs[metadata['source']] = ([], [])
                        all_docs[metadata['source']][0].append(doc_id)
                        all_docs[metadata['source']][1].append(doc_data)
                print()

    for doc_source, (doc_ids, doc_chunks) in all_docs.items():
        r = []
        for need in needed_info:
            if need == 'source':
                r.append(doc_source)
            elif need == 'chunks':
                r.append(len(doc_chunks))
            elif need == 'size':
                r.append(sum(len(x) for x in doc_chunks))
        response.append(r)
    return response

def vs_collection_clear(vs: str):
    vstore = ALL_VECTORSTORES.get(vs)
    if vstore is None:
        return
    vstore.delete()


# LLM utils

def get_llm() -> BaseLLM:
    return OpenAI(
        max_tokens=256, # 4096
        streaming=True,
        temperature=0,
        verbose=True
    )


def load_chain_module_from_path(chain_path: str, perform_reload: bool = False):
    from importlib import import_module, reload
    module_name, chain_var = chain_path.rsplit(':', 1)
    module = import_module(module_name, '.')
    if perform_reload:
        module = reload(module)
    return module, chain_var


def load_chain(chain_path: str, llm: BaseLLM, perform_reload: bool = False, progress = gr.Progress()) -> Tuple[str, Chain]:
    load_iter = progress.tqdm(range(3), desc="Loading")
    if isinstance(chain_path, (str, list)):
        if isinstance(chain_path, list) and len(chain_path) > 0:
            chain_path = chain_path[0]
        if len(chain_path) == 0:
            raise ValueError("Chain path is empty!")

        chain_args = {}
        chain_args['llm'] = llm

        if True:
            chain_args['retriever'] = ALL_VECTORSTORES['genesisdb'].as_retriever(
                search_kwargs={"k": TARGET_SOURCE_CHUNKS},
                search_type='similarity',
            )

        load_iter.update()
        chain_module, chain_var = load_chain_module_from_path(chain_path, perform_reload)
        load_iter.update()
        chain_factory: Callable[[BaseLLM], Chain] = getattr(chain_module, chain_var)
        try:
            chain_obj = chain_factory(**chain_args)
        except TypeError:
            print("Chain %s doesn't support retrievers. Disabling." % chain_path)
            chain_args.pop('retriever', None)
            chain_obj = chain_factory(**chain_args)
        load_iter.update()
        return chain_path, chain_obj

    raise ValueError("Unable to load the chain \"%s\"" % str(chain_path))

def reload_chain(chain_path: str, llm: BaseLLM):
    return load_chain(chain_path, llm, perform_reload=True)


def get_uploaded_files_list(vs: str) -> pd.DataFrame:
    return pd.DataFrame(vs_doc_list(vs, "source", "chunks", "size"), columns=["source", "chunks", "size (characters)"])

def clear_collection(vs: str, progress=gr.Progress()):
    vs_collection_clear(vs)
    return get_uploaded_files_list(vs)

async def handle_upload(files: List[_TemporaryFileWrapper], vs: str, progress=gr.Progress()):
    # Note: The _TemporaryFileWrapper is only useful for getting the filename as it has no access to its content
    from doc_ingest import upload_files
    vstore = ALL_VECTORSTORES.get(vs)
    upload_task = asyncio.get_event_loop().run_in_executor(None, upload_files, vstore, *files)
    new_uploaded_files = await upload_task
    return get_uploaded_files_list(vs)


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
    loaded_llm = gr.State(get_llm)
    loaded_chain = gr.State()
    prepared_message = gr.State('')

    with gr.Row(equal_height=False):
        gr.Markdown("# PrivateGPT Demo")

    with gr.Row():
        tb_chain_path = gr.Textbox(
            label="Chain factory function",
            placeholder="Path to the chain module",
            show_label=True,
            lines=1,
            type="text",
            scale=3
        )
        with gr.Column(scale=1):
            load_chain_btn = gr.Button("Load chain", size='sm')
            reload_chain_btn = gr.Button("Reload chain", size='sm')

        example_dataset = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=CHAIN_EXAMPLES,
            label="Preset chain",
            type="values",
            samples_per_page=3
        )

    chat_engine = ChatWrapper()
    chatbot = gr.Chatbot()

    with gr.Column():
        with gr.Row():
            txt_message = gr.Textbox(
                label="Chat",
                placeholder="Example: Describe the contents",
                lines=1,
                scale=11,
                show_label=False
            )
            submit = gr.Button(value="Send", variant="secondary", scale=1)

        with gr.Row(variant="compact"):
            rd_output_mode = gr.Radio(
                [LLMOutputMode.DIRECT, LLMOutputMode.STREAM],
                value=LLMOutputMode.DIRECT,
                label="Output mode",
                interactive=True
            )
            dd_select_vs = gr.Dropdown(
                choices=ALL_VECTORSTORES.keys(),
                value=list(ALL_VECTORSTORES.keys())[0],
                label="Vector Store",
                show_label=True,
                allow_custom_value=False,
                type="value"
            )

    with gr.Row():
        file_upload_box = gr.File(file_count="multiple", label="Upload files", show_label=True, interactive=True)
        with gr.Column():
            list_files_ready = gr.DataFrame(
                value=lambda: get_uploaded_files_list(dd_select_vs.value),
                type="pandas",
                label="Documents in collection",
                interactive=False
            )
            btn_clear_col = gr.Button("Clear entire collection", variant="stop", size="sm")

    gr.HTML("Demo application of a Langchain-based PrivateGPT.")

    clr_msg_box = lambda: ''

    submit.click(chat_engine.prepare, inputs=[txt_message, chatbot, tb_chain_path, loaded_chain], outputs=[chatbot, prepared_message]) \
        .success(clr_msg_box, outputs=txt_message) \
        .success(chat_engine.generate, inputs=[prepared_message, chatbot, rd_output_mode, loaded_chain], outputs=chatbot)

    txt_message.submit(chat_engine.prepare, inputs=[txt_message, chatbot, tb_chain_path, loaded_chain], outputs=[chatbot, prepared_message]) \
        .success(clr_msg_box, outputs=txt_message) \
        .success(chat_engine.generate, inputs=[prepared_message, chatbot, rd_output_mode, loaded_chain], outputs=chatbot)

    example_dataset.select(load_chain, inputs=[example_dataset, loaded_llm], outputs=[tb_chain_path, loaded_chain],
        show_progress='minimal')
    load_chain_btn.click(load_chain, inputs=[tb_chain_path, loaded_llm], outputs=[tb_chain_path, loaded_chain],
        show_progress='minimal')
    reload_chain_btn.click(reload_chain, inputs=[tb_chain_path, loaded_llm], outputs=[tb_chain_path, loaded_chain],
        show_progress='minimal')

    dd_select_vs.select(get_uploaded_files_list, inputs=dd_select_vs, outputs=list_files_ready)
    file_upload_box.upload(handle_upload, inputs=[file_upload_box, dd_select_vs], outputs=list_files_ready, show_progress='minimal')
    btn_clear_col.click(clear_collection, inputs=dd_select_vs, outputs=list_files_ready)

if __name__ == "__main__":
    demo.launch(debug=True)
