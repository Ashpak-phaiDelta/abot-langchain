
import json
import pandas as pd

from .base import create_api_tool

from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools.base import Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import OpenAPIEndpointChain
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain

from ..prompts import is_chat_model, get_agent_prompt, GENESIS_UNIT_LEVEL_AGENT_PROMPT_PREFIX

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


def get_unit_level_query_agent(llm, llm_for_tool, spec, requests, memory, verbose: bool = False) -> AgentExecutor:
    return initialize_agent(
        [
            get_tool_genesis_unit_sensor_list(llm_for_tool, spec, requests, verbose),
            Tool(
                func=lambda q: "Error: Need to know warehouse name",
                name="get_unit_status",
                description="Use when status of a unit within a warehouse is needed, such as \"B2 Basement status\", \"Cipla unit information\", etc. It gives status such as how many sensors are normal/inactive/out, unit state."
            )
        ],
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION if is_chat_model(llm) else AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=verbose,
        memory=memory,
        agent_kwargs=get_agent_prompt(llm,
            prefix_prompt=GENESIS_UNIT_LEVEL_AGENT_PROMPT_PREFIX
        )
    )


__all__ = [
    'get_tool_genesis_unit_sensor_list',
    'get_unit_level_query_agent'
]
