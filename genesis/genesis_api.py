import json
import gzip
from transformers import GPT2TokenizerFast
import yaml


from langchain.chains import OpenAPIEndpointChain, LLMChain
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.requests import Requests
from langchain.agents.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec

from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.agents import create_json_agent, AgentExecutor
from langchain.llms.openai import OpenAI

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
                     output_processor: Callable[[], str] = None):
    api_operation = APIOperation.from_openapi_spec(spec, endpoint, method)
    chain = OpenAPIEndpointChain.from_api_operation(
        api_operation,
        llm,
        requests=requests,
        verbose=TOOL_VERBOSE,
        return_intermediate_steps=False,
        raw_response=not auto_parse_output_using_llm
    )
    tool_func = chain.run
    if output_processor is not None:
        tool_func = lambda txt, **kwargs: output_processor(context=chain.run(txt, **kwargs), query=txt)
    return Tool(
        name=name or api_operation.operation_id,
        description=description or api_operation.description,
        func=tool_func
    )

def reduce_tokens(input_text, max_tokens):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenized_input = tokenizer.encode(input_text, add_special_tokens=False)
    
    if len(tokenized_input) <= max_tokens:
        return input_text
    
    reduced_tokens = tokenized_input[:max_tokens]
    reduced_text = tokenizer.decode(reduced_tokens)
    
    return reduced_text


def _get_tool_genesis_sensor_status(llm, spec, requests):
    return _create_api_tool(
        llm, spec, requests,
        '/genesis/query/sensor',
        description='Use to get sensor details, metadata, status and location information from given sensor ID. Do not use if sensor type or name is given, use another tool. Do not make up an answer if this tool fails.'
    )


def _get_tool_genesis_sensor_list(llm, spec, requests):
    def process_chain_output(context: str, query: str) -> str:
        data = json.loads(context)
        reply = llm(f"List all keys from this json {json.dumps(data[0])}")



        # key = llm(f"where would {query} fit as value from this list of keys [{reply}] ")
        key = llm(f"""
                  Instructions:
                  - Select best suited key from given list [{reply}]
                  Example: 
                  - Temperature :-> sensor_type
                  - VER_W2_B5_FF_C : -> global_unit_name
                  - Humidity :-> sensor_type
                  - KUD_W1_B2_GF_X :-> global_unit_name

                Given the following text:
                Question: {query}
                Answer:

                  """)

        print(key, query)

        list_of_sensor= []
        # context = json.loads(context)
        for sensor in data:
            if sensor[key.strip()].lower() == query.strip().lower():
                list_of_sensor.append({
                        "sensor_id": sensor['sensor_id'],
                        "sensor_name": sensor['global_sensor_name'],
                        "sensor_type": sensor['sensor_type'],
                        "unit_id": sensor['unit_id'],
                        "unit_name": sensor['global_unit_name']
                        }
            )
        context = json.dumps(list_of_sensor)
        print(context)


        # list_of_item = []
        # for item in data:
        #     flag = llm("reply in 1 or 0 if {query} fits for {item}")
        #     print(".",end="")
        #     if int(flag):
        #         list_of_item.append(item)

        # print(item)

        return context

    return _create_api_tool(
        llm, spec, requests,
        '/sensors',
        name='sensor_list_to_get_sensor_id',
        description='Use to get a list of every sensor available in the whole application. their id (integer for use in summary), name and status of sensor (normal, out_of_range, etc) ID can be used for other operations that need it. tool can be used to get ID for other tools',
        output_processor=process_chain_output
        
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
    def process_chain_output(context: str, query: str) -> str:
        print("post-process output")
        print('--------')
        print(":::Context:::")
        print(context)
        print(':::Query:::')
        print(query)
        print('--------')
        return '3'
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
