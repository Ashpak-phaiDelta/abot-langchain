
from .base import create_api_tool


def get_tool_genesis_sensor_status(llm, spec, requests, verbose: bool = False):
    return create_api_tool(
        llm, spec, requests,
        '/metrics/warehouse/{warehouse_id}',
        name="warehouse_sensor_summary",
        description='''Use to get the status of a sensor such as its value, state, duration of state, etc.''',
        verbose=verbose
    )


__all__ = [
    'get_tool_genesis_sensor_status'
]
