
import json
import pandas as pd

from langchain.chains import OpenAPIEndpointChain, LLMChain
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.requests import Requests
from langchain.tools.base import Tool, StructuredTool
from langchain.prompts import PromptTemplate
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec

from pydantic import BaseModel
from typing import Optional, Callable


def _create_api_tool(llm: BaseLanguageModel,
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

# Old, from fulfillment
# def _get_tool_genesis_sensor_status(llm, spec, requests):
#     return _create_api_tool(
#         llm, spec, requests,
#         '/genesis/query/sensor',
#         description='Use to get sensor details, metadata, status and location information from given sensor ID. Do not use if sensor type or name is given, use another tool. Do not make up an answer if this tool fails.'
#     )


def get_tool_genesis_sensor_list(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        def _process_request(original_query: str):
            # TODO Finish this
            response_data = chain.run({})

            data = json.loads(response_data)
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
                    Question: {original_query}
                    Answer:

                    """)

            print(key, original_query)

            list_of_sensor= []
            # context = json.loads(context)
            for sensor in data:
                if sensor[key.strip()].lower() == original_query.strip().lower():
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
        return _process_request
    return _create_api_tool(
        llm, spec, requests,
        '/sensors',
        name='sensor_list_to_get_sensor_id',
        description='Use to get a list of every sensor available in the whole application. their id (integer for use in summary), name and status of sensor (normal, out_of_range, etc) ID can be used for other operations that need it. tool can be used to get ID for other tools, Input MUST have the original query, and no other parameters are there. It MUST be a dictionary such as {{"original query": "what the user typed"}}, no filters are there.',
        verbose=verbose,
        output_processor=process_chain_output
    )


def get_tool_genesis_location_list(llm, spec, requests, verbose: bool = False):
    return _create_api_tool(
        llm, spec, requests,
        '/locations',
        name='location_list_all_location_names',
        description='Use to get a list of all locations/warehouses, their `id` (integer for use in summary), name and status of the location (normal, out_of_range, etc). Use it to find location ID given its name by filtering it. The ID can be used for other operations that need it. This tool can be used to get ID for other tools. This tool requires no input parameters, but an empty "" string.',
        verbose=verbose,
    )

def get_tool_genesis_location_summary(llm, spec, requests, verbose: bool = False):
    return _create_api_tool(
        llm, spec, requests,
        '/locations/{id}/summary',
        name='location_summary_and_status',
        description='Can get summary of a location/warehouse id (eg: id=1, counter-example: id=VER_W1) (ONLY integer, not like VER_W1 or Verna, etc, but MUST be like 1, 2, etc.) such as power, attendance, metrics summary, emergency, etc. Use `location_list_all_location_names` tool to get id first if not known.',
        verbose=verbose
    )


def get_tool_genesis_warehouse_summary(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        class ParamModel(BaseModel):
            original_query: str
            location_id: int
        def warehouse_sensor_summary(query: str) -> str:
            schema = ParamModel.parse_obj(json.loads(query))
            params_jsonified = json.dumps({
                "warehouse_id": schema.location_id
            })
            response_data = chain.run(params_jsonified)
            resp_json = json.loads(response_data)

            response_text_summary = ''

            response_text_summary += '# Warehouse level sensors\n'
            warlvl_sensors = resp_json['wv_warehouse_metrics']

            warlvl_df = pd.DataFrame(warlvl_sensors)

            response_text_summary += "Level info: ## is Sensor type, ### is Sensor subtype\n"
            response_text_summary += "Sensor data format: Name: Value: Status\n"

            for metric_type_name, df_mt in warlvl_df.groupby('Metric Type'):
                response_text_summary += "## %s\n" % metric_type_name
                for metric_subtype_name, df_mst in df_mt.groupby('Metric Sub-Type'):
                    response_text_summary += "### %s\n" % metric_subtype_name
                    for lbl, sensor_row in df_mst.iterrows():
                        val = sensor_row['Value'] or ''
                        if sensor_row['Unit'] is not None:
                            val += ' ' + sensor_row['Unit']
                        if sensor_row['Value Duration Minutes'] is not None:
                            val += '(for %s)' % sensor_row['Value Duration Minutes']
                        response_text_summary += '{}: {}: {}\n'.format(
                            sensor_row['Sensor Name'],
                            val,
                            sensor_row['State']
                        )

            # TODO warehouse units
            # response_text_summary += '# Warehouse level units\n'

            return response_text_summary
        return warehouse_sensor_summary
    return _create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{id}',
        name="warehouse_sensor_summary",
        description='''Use to get a summary of all sensors at warehouse-level/location given the `location_id` value. No unit-level sensors. It can give a list of sensors in the warehouse-level, their values, state, etc. Count each sensor's status for the question 'How many sensors are out_of_range?'. You can infer `location_id` from previous input, else ask user to enter warehouse name. Following parameters are REQUIRED, passed as valid stringified-json:
{{"original_query": string - The query user had given, "location_id": integer - the ID (1,2,etc) of the location/warehouse that the user requested. If not known, ask user for which warehouse}}''',
        verbose=verbose,
        output_processor=process_chain_output
    )

def get_tool_genesis_warehouse_unit_summary(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        class ParamModel(BaseModel):
            original_query: str
            location_id: int
        def warehouse_unit_summary(query: str) -> str:
            schema = ParamModel.parse_obj(json.loads(query))
            params_jsonified = json.dumps({
                "warehouse_id": schema.location_id
            })
            response_data = chain.run(params_jsonified)
            resp_json = json.loads(response_data)

            response_text_summary = ''
            
            response_text_summary += '# Warehouse-level units\n'
            warlvl_sensors = resp_json['wv_unit_summary']

            warlvl_df = pd.DataFrame(warlvl_sensors)

            response_text_summary += "> Data format: Name (alias): Out-count: Status\n"

            for lbl, unit_row in warlvl_df.iterrows():
                name = unit_row['Unit Name'] or ''
                if unit_row['Unit Alias'] is not None:
                    name += '(%s)' % unit_row['Unit Alias']
                response_text_summary += '{}: {}: {}\n'.format(
                    name,
                    unit_row['Value'],
                    unit_row['State']
                )

            return response_text_summary
        return warehouse_unit_summary
    return _create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{id}',
        name='warehouse_unit_summary',
        description='''Use to get a summary of all units at warehouse-level/location given the `location_id` value. It can give a list of units in the warehouse-level, count of out_of_range sensors in it, state, etc. eg: 'How many sensors are out_of_range in unit X?'. You can infer `location_id` from previous input, else ask user to enter warehouse name. Following parameters are REQUIRED, passed as valid stringified-json:
{{"original_query": string - The query user had given, "location_id": integer - the ID (1,2,etc) of the location/warehouse that the user requested. If not known, ask user for which warehouse}}''',
        verbose=verbose,
        output_processor=process_chain_output
    )


__all__ = [
    # '_get_tool_genesis_sensor_status',
    'get_tool_genesis_sensor_list',
    'get_tool_genesis_location_summary',
    'get_tool_genesis_location_list',
    'get_tool_genesis_warehouse_summary',
    'get_tool_genesis_warehouse_unit_summary'
]
