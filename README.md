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
