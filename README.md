# Abot Langchain

## Development

1. Install dependencies

```shell
virtualenv venv
# Linux
source venv/bin/activate
# Windows
./venv/Script/activate

pip install -r requirements.txt -r requirements.deploy.txt
```

2. Run the Genesis Agent with a shell interface

```shell
python -m genesis
```

__(Ctrl-C to exit)__


## Gradio

1. Install `gradio`:

```shell
pip install gradio
```

2. Launch the blocks:

```shell
gradio langchain_gradio.py
```

3. Open the landing page in browser: [http://localhost:7860/](http://localhost:7860/)

4. Select a chain from the examples listed, or type the path to it (package.module:chain_factory_fn format). Eg: `doc_parse:ask_doc_chain`. The LLM is passed as the keyword argument `llm` to that function.

5. Begin asking questions

6. You can upload a file below the chatbot, it will be ingested automatically
