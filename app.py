
from fastapi import FastAPI

from langcorn import create_service


app: FastAPI = create_service("genesis.chat_chain:agent_chain")
