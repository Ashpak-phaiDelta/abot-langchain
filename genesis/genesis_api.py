
import json

from langchain.chains import OpenAPIEndpointChain, LLMChain
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.requests import Requests
from langchain.tools.base import Tool, StructuredTool
from langchain.prompts import PromptTemplate
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec

from typing import Optional, Callable


TOOL_VERBOSE = False

def _create_api_tool(llm: BaseLanguageModel,
                     spec: OpenAPISpec,
                     requests: Requests,
                     endpoint: str,
                     method: str = 'get',
                     name: Optional[str] = None,
                     description: Optional[str] = None,
                     auto_parse_output_using_llm: bool = False,
                     output_processor: Optional[Callable[[Chain], Callable[..., str]]] = None):
    api_operation = APIOperation.from_openapi_spec(spec, endpoint, method)
    chain = OpenAPIEndpointChain.from_api_operation(
        api_operation,
        llm,
        requests=requests,
        verbose=TOOL_VERBOSE,
        return_intermediate_steps=False,
        raw_response=not auto_parse_output_using_llm
    )
    
    if output_processor is not None:
        return StructuredTool.from_function(
            func=output_processor(chain),
            name=name or api_operation.operation_id,
            description=description or api_operation.description
        )

    return Tool(
        name=name or api_operation.operation_id,
        description=description or api_operation.description,
        func=chain.run
    )



def _get_tool_genesis_sensor_status(llm, spec, requests):
    return _create_api_tool(
        llm, spec, requests,
        '/genesis/query/sensor',
        description='Use to get sensor details, metadata, status and location information from given sensor ID. Do not use if sensor type or name is given, use another tool. Do not make up an answer if this tool fails.'
    )


def _get_tool_genesis_sensor_list(llm, spec, requests):
    return _create_api_tool(
        llm, spec, requests,
        '/sensors',
        description='Use to get a list of every sensor available in the whole application'
    )


def get_tool_genesis_location_list(llm, spec, requests):
    return _create_api_tool(
        llm, spec, requests,
        '/locations',
        name='location_list_to_get_location_id',
        description='Use to get a list of all locations/warehouses, their `id` (integer for use in summary), name and status of the location (normal, out_of_range, etc). Use it to find location ID given its name by filtering it. The ID can be used for other operations that need it. This tool can be used to get ID for other tools.'
    )

def get_tool_genesis_location_summary(llm, spec, requests):
    return _create_api_tool(
        llm, spec, requests,
        '/locations/{id}/summary',
        name='location_summary_id_integer',
        description='Can get summary of a location/warehouse id (eg: /locations/1/summary) (counter-example: /locations/VER_W1/summary) (ONLY integer, not like VER_W1 or Verna, etc, but MUST be like 1, 2, etc.) such as power, attendance, metrics summary, etc. Use `location_list` tool to get id first'
    )
    

def get_tool_genesis_warehouse_summary(llm, spec, requests):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        def _process_request(original_query: str, warehouse_id: int):
            params_jsonified = json.dumps({
                "warehouse_id": warehouse_id
            })
            response_data = chain.run(params_jsonified)
            print("processor output")
            print('--------')
            print(":::Parameters:::")
            print(params_jsonified)
            print(':::Query:::')
            print(original_query)
            print(':::Response:::')
            print(response_data)
            print('--------')
            
            # Stub response. Replace after searching
            return "Can't find requested details"
        return _process_request
    return _create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{id}',
        description="Use to get a summary of all sensors at warehouse-level id, i.e. inside location. (eg: /metrics/warehouse/1). Counter-example: /locations/VER_W1/summary. It can give a list of sensors in the warehouse-level, their values, state, etc. Use it to also get a list of units and their status. i.e. How many sensors in each unit are out of range/normal, or count each unit's status for the question 'How many units are out_of_range?'.",
        output_processor=process_chain_output
    )

__all__ = [
    '_get_tool_genesis_sensor_status',
    '_get_tool_genesis_sensor_list',
    'get_tool_genesis_location_summary',
    'get_tool_genesis_location_list',
    'get_tool_genesis_warehouse_summary'
]
