
import json
import pandas as pd

from .base import create_api_tool

from langchain.chains import OpenAPIEndpointChain
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain

from pydantic import BaseModel
from typing import Callable


def get_tool_genesis_unit_sensor_list(llm, spec, requests, verbose: bool = False):
    def process_chain_output(chain: OpenAPIEndpointChain) -> Callable[..., str]:
        class ParamModel(BaseModel):
            original_query: str
            location_id: int
            unit_id: int
        def unit_sensor_summary(query: str) -> str:
            schema = ParamModel.parse_obj(json.loads(query))
            params_jsonified = json.dumps({
                "warehouse_id": schema.location_id,
                "unit_id": schema.unit_id
            })
            response_data = chain.run(params_jsonified)
            resp_json = json.loads(response_data)

            response_text_summary = ''
            
            response_text_summary += '# Unit-level sensors\n'
            unitlvl_sensors = resp_json['uv_unit_metrics']

            unitlvl_df = pd.DataFrame(unitlvl_sensors)

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
        description='''Use to get a summary and list of all sensors at unit-level/location given the `location_id` value. It can give a list of sensors in the unit-level (within a warehouse), their ID, state (normal, out_of_range), etc. eg: 'How many sensors are out_of_range in unit X?' or 'How many sensors are in unit Abc?'. You can infer `location_id` and `unit_id` from previous input, else ask user to enter warehouse name. Following parameters are REQUIRED, passed as valid stringified-json:
{{"original_query": str - $The query user had given$, "location_id": int - $the ID (1,2,etc) of the location/warehouse that the user requested. If not known, use another tool to get first, else ask human for which warehouse. Make SURE location_id is correct before using$, "unit_id": int - $the ID (1,2,etc) of the unit that the user requested. If not known, use another tool to get first. Make SURE unit_id is correct before using$}}
The text between $text$ are instructions for you''',
        verbose=verbose,
        output_processor=process_chain_output
    )


__all__ = [
    'get_tool_genesis_unit_sensor_list'
]
