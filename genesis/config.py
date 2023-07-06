
from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, AnyUrl, FilePath
from typing import Union

from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec


class GenesisSettings(BaseSettings):
    auth_token: str
    agent_is_verbose: bool = False
    tool_is_verbose: bool = False
    openapi_file: Union[AnyUrl, FilePath, str] = 'genesis_openapi.yaml'

    class Config:
        env_file = '.env'
        env_prefix = 'genesis_'


# Settings fetchers

@lru_cache()
def get_auth_token():
    return GenesisSettings().auth_token

@lru_cache()
def get_agent_is_verbose():
    return GenesisSettings().agent_is_verbose

@lru_cache()
def get_tool_is_verbose():
    return GenesisSettings().tool_is_verbose

@lru_cache()
def get_openapi_file():
    return GenesisSettings().openapi_file


def fetch_genesis_spec() -> OpenAPISpec:
    spec_file = get_openapi_file()

    if isinstance(spec_file, AnyUrl):
        return OpenAPISpec.from_url(spec_file)
    elif Path(spec_file).exists():
        return OpenAPISpec.from_file(spec_file)

    raise ValueError("You must set the setting `openapi_file` or `GENESIS_OPENAPI_FILE` environment to a path that exists.\nIt was set to '%s'" % str(spec_file))

