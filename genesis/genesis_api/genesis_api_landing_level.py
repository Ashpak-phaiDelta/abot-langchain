
from .base import create_api_tool


def get_tool_genesis_location_list(llm, spec, requests, verbose: bool = False):
    return create_api_tool(
        llm, spec, requests,
        '/locations',
        name='location_list_all_location_names',
        description='Use to get a list of all locations/warehouses, their warehouse_id "id" (integer for use in summary), name and status of the location (normal, out_of_range, etc). Use it to find location ID given its name by filtering it. The ID can be used for other operations that need it. This tool can be used to get ID for other tools. This tool requires no input parameters, but an empty "" string. Key "id" is same as "warehouse_id".',
        verbose=verbose,
    )

def get_tool_genesis_location_summary(llm, spec, requests, verbose: bool = False):
    return create_api_tool(
        llm, spec, requests,
        '/locations/{warehouse_id}/summary',
        name='location_summary_and_status',
        description='Can get summary of a location/warehouse id (eg: warehouse_id=1, counter-example: warehouse_id=VER_W1) (ONLY integer, not like VER_W1 or Verna, etc, but MUST be like 1, 2, etc.) such as power, attendance, metrics summary, emergency, etc. Use `location_list_all_location_names` tool to get id first if not known.',
        verbose=verbose
    )


__all__ = [
    'get_tool_genesis_location_summary',
    'get_tool_genesis_location_list'
]
