
import json
import pandas as pd

from .base import create_api_tool

from langchain.chains import OpenAPIEndpointChain
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain

from pydantic import BaseModel
from typing import Callable


def get_tool_genesis_warehouse_summary(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        class ParamModel(BaseModel):
            original_query: str
            location_id: int
        def warehouse_sensor_summary(query: str) -> str:
            # schema = ParamModel.parse_obj(json.loads(query))
            # params_jsonified = json.dumps({
            #     "warehouse_id": schema.location_id
            # })
            try:
                response_data = chain.run(query)
                resp_json = json.loads(response_data)
            except:
                return 'Error making request. Try again in some time.'

            response_text_summary = ''

            response_text_summary += '# Warehouse level sensors\n'
            warlvl_sensors = resp_json['wv_warehouse_metrics']

            warlvl_df = pd.DataFrame(warlvl_sensors)
            if len(warlvl_df) == 0:
                return 'No sensors are present in this warehouse'

            response_text_summary += "Level info: ## is Sensor type, ### is Sensor subtype\n"
            response_text_summary += "Sensor data format: Sensor ID // Name // Value // Status\n"

            for metric_type_name, df_mt in warlvl_df.groupby('Metric Type'):
                response_text_summary += "## %s\n" % metric_type_name
                for metric_subtype_name, df_mst in df_mt.groupby('Metric Sub-Type'):
                    response_text_summary += "### %s\n" % metric_subtype_name
                    for lbl, sensor_row in df_mst.iterrows():
                        name = sensor_row['Sensor Name']
                        val = sensor_row['Value'] or ''
                        # Display unit (celcius, %, etc.)
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
        return warehouse_sensor_summary
    return create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{warehouse_id}',
        name="warehouse_sensor_summary",
        description='''Use to get a summary of all sensors at warehouse-level/location given the `location_id` value. Can be use to search a sensor from name. No unit-level sensors. It can give a list of sensors in the warehouse-level, their ID, their values, state, etc. Count each sensor's status for the question 'How many sensors are out_of_range?'. You can infer `location_id` from tool "location_list_all_location_names", else ask user to enter warehouse name. Following parameters are REQUIRED, passed as valid stringified-json:
{{"original_query": str - $The query user had given$, "location_id": int - $the ID (1,2,etc) of the location/warehouse that the user requested. If not known, ask human for which warehouse. Make SURE location_id is correct data type and value before using$}}
The text between $text$ are instructions for you''',
        verbose=verbose,
        output_processor=process_chain_output
    )

def get_tool_genesis_warehouse_unit_summary(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        class ParamModel(BaseModel):
            original_query: str
            location_id: int
        def warehouse_unit_list_summary(query: str) -> str:
            # schema = ParamModel.parse_obj(json.loads(query))
            # params_jsonified = json.dumps({
            #     "warehouse_id": schema.location_id
            # })
            try:
                response_data = chain.run(query)
                resp_json = json.loads(response_data)
            except:
                return 'Error making request. Try again in some time.'

            response_text_summary = ''
            
            response_text_summary += '# Warehouse-level units\n'
            warlvl_sensors = resp_json['wv_unit_summary']

            warlvl_df = pd.DataFrame(warlvl_sensors)

            if len(warlvl_df) == 0:
                return 'No units are present in this warehouse'

            response_text_summary += "> Data format: Unit ID // Name (alias) // Number of out_of_range sensors // Status\n"

            for lbl, unit_row in warlvl_df.iterrows():
                name = unit_row['Unit Name'] or ''
                if unit_row['Unit Alias'] is not None:
                    name += '(%s)' % unit_row['Unit Alias']
                response_text_summary += '{} // {} // {} // {}\n'.format(
                    unit_row['Unit Id'],
                    name,
                    unit_row['Value'],
                    unit_row['State']
                )

            return response_text_summary
        return warehouse_unit_list_summary
    return create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{warehouse_id}',
        name='warehouse_unit_list_summary',
        description='''Use to get a summary of all units at warehouse-level/location given the `location_id` value. For a list of sensors, use a different tool. It can give a list of units in the warehouse-level, their unit id, count of out_of_range sensors in it, state, etc. eg: 'How many sensors are out_of_range in unit X?'. You can infer `location_id` from tool "location_list_all_location_names", else ask user to enter warehouse name. Following parameters are REQUIRED, passed as valid stringified-json:
{{"original_query": str - $the query user_had given$, "location_id": int - $the ID (1,2,etc) of the location/warehouse that the user requested. If not known, ask human for which warehouse. Make SURE location_id is correct before using by fetching$}}
The text between $text$ are instructions for you''',
        verbose=verbose,
        output_processor=process_chain_output
    )


__all__ = [
    'get_tool_genesis_warehouse_summary',
    'get_tool_genesis_warehouse_unit_summary'
]
