
from fastapi import FastAPI

from langcorn import create_service
from langcorn.server import api as lcorn_api

# Monkey-patching to fix bug of input/output keys (till fix is updated)
def _fixed_derive_fields(language_app):
    if hasattr(language_app, "input_variables"):
        return language_app.input_variables, language_app.output_variables
    elif hasattr(language_app, "prompt"):
        return language_app.prompt.input_variables, [language_app.output_key]
    return [x for x in language_app.input_keys if x != 'chat_history'], language_app.output_keys

setattr(lcorn_api, 'derive_fields', _fixed_derive_fields)


app: FastAPI = create_service("genesis.chat_chain:agent_chain")
