
from langchain.chains import OpenAPIEndpointChain
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.requests import Requests
from langchain.tools.base import Tool
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec

from typing import Optional, Callable


def create_api_tool(llm: BaseLanguageModel,
                     spec: OpenAPISpec,
                     requests: Requests,
                     endpoint: str,
                     method: str = 'get',
                     name: Optional[str] = None,
                     description: Optional[str] = None,
                     verbose: bool = False,
                     auto_parse_output_using_llm: bool = False,
                     output_processor: Optional[Callable[[Chain], Callable[..., str]]] = None):
    api_operation = APIOperation.from_openapi_spec(spec, endpoint, method)
    chain = OpenAPIEndpointChain.from_api_operation(
        api_operation,
        llm,
        requests=requests,
        verbose=verbose,
        return_intermediate_steps=False,
        raw_response=not auto_parse_output_using_llm,
    )

    if output_processor is not None:
        return Tool.from_function(
            func=output_processor(chain),
            name=name or api_operation.operation_id,
            description=description or api_operation.description,
            verbose=verbose
        )

    return Tool(
        name=name or api_operation.operation_id,
        description=description or api_operation.description,
        func=chain.run,
        verbose=verbose
    )
