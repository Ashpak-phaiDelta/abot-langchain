
from langchain.tools.base import Tool
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain


def get_tool_genesis_unit_search(llm, spec, requests, verbose: bool = False):
    api_operation = APIOperation.from_openapi_spec(spec, '/metrics/warehouse/{warehouse_id}', 'get')
    unit_search_chain = OpenAPIEndpointChain.from_api_operation(
        api_operation,
        llm,
        requests=requests,
        verbose=verbose,
        return_intermediate_steps=False,
        raw_response=True
    )
    def unit_search(query: str) -> str:
        print("Query:", query)
        response = unit_search_chain.run("Find the unit with name or alias as \"%s\"" % query.replace('"', "'"))
        print("Response:", response)
        return 'Not found'

    return Tool.from_function(
        func=unit_search,
        name='unit_search',
        description='''Use to find the id of a unit from the unit's name. For example, unit named "Cipla" can return id 1000, "B2 Basement" can be 1001, etc. This ID is used for other tools. Provide input only of the Unit name to search for, for example, "Cipla", "B2 Basement", and will return the id of the unit and warehouse, else "Not found".
Response will be a JSON in the following format:
{{"unit_id": $id of the unit found, eg. 1$, "warehouse_id": $id of the warehouse this unit belongs to$}}''',
        verbose=verbose
    )


__all__ = [
    'get_tool_genesis_unit_search'
]
