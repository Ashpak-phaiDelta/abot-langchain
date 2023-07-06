
import json
import pandas as pd

from .base import create_api_tool

from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools.base import Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import OpenAPIEndpointChain
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain

from pydantic import BaseModel
from typing import Callable


def get_tool_genesis_unit_sensor_list(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        def unit_sensor_summary(query: str) -> str:
            try:
                response_data = chain.run(query)
                resp_json = json.loads(response_data)
            except:
                return 'Error making request. Try again in some time.'

            response_text_summary = ''
            
            response_text_summary += '# Unit-level sensors\n'
            unitlvl_sensors = resp_json['uv_unit_metrics']

            unitlvl_df = pd.DataFrame(unitlvl_sensors)
            if len(unitlvl_df) == 0:
                return 'No metrics are present in this unit'

            response_text_summary += "Level info: ## is Sensor type, ### is Sensor subtype\n"
            response_text_summary += "Sensor data format: Sensor ID // Name // Value // Status\n"

            for metric_type_name, df_mt in unitlvl_df.groupby('Metric Type'):
                response_text_summary += "## %s\n" % metric_type_name
                for metric_subtype_name, df_mst in df_mt.groupby('Metric Sub-Type'):
                    response_text_summary += "### %s\n" % metric_subtype_name
                    for lbl, sensor_row in df_mst.iterrows():
                        name = sensor_row['Sensor Name']
                        val = sensor_row['Value'] or ''
                        if sensor_row['Unit'] is not None:
                            val += ' ' + sensor_row['Unit']
                        if sensor_row['Value Duration Minutes'] is not None:
                            val += '(for %s)' % sensor_row['Value Duration Minutes']
                        response_text_summary += '{} // {} // {} // {}\n'.format(
                            sensor_row['Sensor Id'],
                            name,
                            val,
                            sensor_row['State']
                        )

            return response_text_summary
        return unit_sensor_summary
    return create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{warehouse_id}/unit/{unit_id}',
        name='unit_sensor_summary',
        description='''Use to get a summary and list of all sensors at unit-level/location given the `warehouse_id` and `unit_id` value, not the name. Provide input as valid stringified JSON with the parameters. eg: "{{\\"warehouse_id\\": 1}}"''',
        verbose=verbose,
        output_processor=process_chain_output
    )



def get_tool_genesis_unit_search(llm, spec, requests, verbose: bool = False):
    def unit_search(query: str) -> str:
        print("Query:", query)
        return 'Not found'
    return Tool.from_function(
        func=unit_search,
        name='unit_search',
        description='''Use to find the id of a unit from the unit's name. For example, unit named "Cipla" can return id 1000, "B2 Basement" can be 1001, etc. This ID is used for other tools. Provide input only as valid stringified json string. Eg: "{{\\"unit_name_query\\": "$name of the unit that user requested$", \\"warehouse\\": $Warehouse name if provided (optional). Omit field if not.$}}". The parts between $ are instructions for you.''',
        verbose=verbose
    )



def get_unit_level_query_agent(llm, llm_for_tool, spec, requests, verbose: bool = False) -> AgentExecutor:
    return initialize_agent(
        [
            get_tool_genesis_unit_sensor_list(llm_for_tool, spec, requests, verbose),
            get_tool_genesis_unit_search(llm_for_tool, spec, requests, verbose)
        ],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs=dict(
            system_message="""You are a powerful API client Assistant that executes the correct APIs from schemas of their parameters from the given query. You are going to be asked about unit-level details or unit-level sensors in an IoT application, which contains warehouses, and units (rooms) are within warehouses. You can access tools that perform the API request and return observation.
For example: `how many units are there?`, `In Cipla unit, which sensors are out?`, `B2 basement status`
Do not make up an answer, if you can't produce the final answer due to no tools satisfying, say "I don't know" and elaborate on what the user could do to improve query.

Make sure to strictly follow "RESPONSE FORMAT INSTRUCTIONS" to produce all output.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist. If you are unable to answer a question, ask user to provide the needed information or say `I don't know`

TOOLS:
------

You have access to the following tools: """, # For Chat-type LLM
        )
    )


__all__ = [
    'get_tool_genesis_unit_sensor_list'
]

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    from langchain.requests import Requests
    from ..basic_llms import llm, llm_tool
    from ..genesis_agent import fetch_genesis_spec, get_auth_token
    from ..utils import read_input

    # Requests with auth token
    requests = Requests(headers={"Authorization": "Bearer %s" % get_auth_token()})

    # Genesis API specifications (OpenAPI)
    spec = fetch_genesis_spec()

    agent = get_unit_level_query_agent(llm, llm_tool, spec, requests, True)
    try:
        while True:
            print(agent.run(read_input()))
    except (KeyboardInterrupt, EOFError):
        # Quit gracefully
        pass
