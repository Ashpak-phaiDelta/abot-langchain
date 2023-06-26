from langchain.agents import Tool
import requests
from typing import Optional
from datetime import datetime, timedelta
import json

from llm import llm


# spec = OpenAPISpec.from_url(
#     "http://uat.phaidelta.com:8079/openapi.json")
# uat = 'http://uat.phaidelta.com:8079/'



def get_response(endpoint):
    url = 'http://localhost:8001'    
    response = requests.get(url+endpoint)
    if response.status_code ==200:
        return response.json()
    else:
        print("response failed")
        return response.status_code


class Sensor():
    def get_sensor_id(self,text) -> int:
        sensor_type = llm(f"""                          
        Instructions:
                1. The purpose of this interaction is to extract sensor types mentioned in the given text.
                2. The extracted sensor types should be returned as a string, with each sensor type capitalized in title case.
                3. The sensor types can include Temperature, Humidity, Motion, Light, Proximity, and Accelerometer.
                4. If no sensor types are found, the response should be an empty string.
                5. Please ensure that the extracted sensor types are accurate and distinct.
                6. Undestand relevant synonyms for the sensor types mentioned above and return as sensor type mentioned above.
                Given the following text:
                        Question:{text} 
                        Answer: """)
        location = llm(f"""
Instructions:
        1. The purpose of this interaction is to extract a location mentioned in the given text.
        2. The extracted location should be returned as a string in capital case.
        3. The location can be in the format 'VER_W1_B2_GF_B' or 'VER_W1_B2_GF_A'.
        4. The location may also contain aliases such as 'Verna Ground floor B' or 'Verna Ground floor A'.
        5. If no location is found, the response should be 'No location found.'
        
        Examples:
        - 'VER_W1_B2_GF_B' corresponds to 'Verna Warehouse 1 Building 2 Ground Floor B'.
        - 'VER_W2_B5_FF_C' corresponds to 'Verna Warehouse 2 Building 5 First Floor C'.
        - 'KUD_W1_B2_GF_X_1_temp' corresponds to 'Kundai Warehouse 1 Building 2 Ground Floor X 1 Temperature'.
        - 'VER_W1_WARLVL_WARLVL_WARLVL' corresponds to 'Verna Warehouse level'.
        - 'KUD_W1_WARLVL_WARLVL_WARLVL' corresponds to 'Kundai Warehouse level'.

                               
Given the following text:
Question: {text}
Answer: 
                    """)
        
        print(f"recovered info from sensor {sensor_type.strip()} and location {location.strip()}")

        sensor_id = self.fetch_sensor_id(sensor_type.strip(), location.strip())
        return sensor_id
    
    def fetch_sensor_id(self,sensor_type : Optional[str] = "", location : Optional[str] = ""):
        response = get_response(f"/genesis/query/sensor/find?sensor_type={sensor_type}&location={location}")
        if isinstance(response, list):
            if len(response)==1:
                return response[0]['sensor_id']
            else:
                sensor_list = []
                for sensor in response:
                    sensor_list.append(sensor['sensor_urn'])
                return sensor_list
        else:
            return f"request failed with status code: {response}"        

    def fetch_sensor_status(self,sensor_id : int):
        response = get_response(f"/genesis/query/sensor_status?sensor_id={sensor_id}")
        try:
            return response['sensor_status']['sensor_health']['code_name']
        except TypeError as e:
            return f"request failed with status code: {response}"

    def get_sensor_status(self,text):
        sensor_id : int = self.get_sensor_id(text)
        if isinstance(sensor_id, int):
            sensor_status : str = self.fetch_sensor_status(sensor_id)
            output =  {"Status":sensor_status}
        elif isinstance(sensor_id, str):
            output = {"Status":"Sensor Not Found"}
        elif isinstance(sensor_id, list):
            output = {"Status": f"Found {len(sensor_id)}No. of sensor here is list {sensor_id}"}
        
        return json.dumps(output)
        
    def get_all_sensor_list(self,text):
        # if text is 'None':
            response = get_response(f"/genesis/query/sensor/list")
            if isinstance(response, list):
                sensor_list = []
                for sensor in response:
                    sensor_list.append(sensor['sensor_urn'])
                return sensor_list
            else:
                return f"request failed with status code: {response}" 
        # else:
        #     sensor_type = llm(f"""Extract sensor type given text and return string containing only sensor type in title case it can include 
        #                             Temperature, Humidity etc.
        #                     Question:{text} 
        #                     Answer: """)
        #     location = llm(f"""Extract location from given text and return string containing only location in Capital case it can include 
        #                             location in format like 'VER_W1_B2_GF_B','VER_W1_B2_GF_A' or it can contain alias of location as 
        #                             'Verna Ground floor B', 'Verna Ground floor A' if no location found return 'No location found'
        #                 Question:{text}
        #                 Answer: 
        #                 """)
        #     if 
        #     print(f"recovered info from sensor {sensor_type} and location {location}")


        #     return "I did not follow"

    


    def get_data_for_sensor(self,text):
        sensor_id = self.get_sensor_id(text)
        print("Got sensor_id : ", sensor_id)
        from_time = llm(f"""
Instructions:
1. The purpose of this interaction is to extract the oldest timestamp mentioned in the given text.
2. The extracted timestamp should be in the format of 'YYYY-MM-DDT00:00:00', such as "today", "tomorrow", or specific dates.
3. The oldest timestamp refers to the earlier datetime.
4. If no timestamps are found, the response should be today's timestamp in 'YYYY-MM-DDT00:00:00' format.
5. The text may contain references to "today" and "tomorrow" for relative timestamps.
6. Other specific dates can also be mentioned, such as "June 1, 2023" or "2023-06-01".
7. Ensure that the extracted timestamp is in a consistent and standardized format for further processing or analysis.

Elaborated Guidelines:
- The text may include sentences or phrases related to timestamps, such as "Give me the data from yesterday", "I need a report starting from June 1, 2023", or "What was the value on January 15, 2022?".
- Relative references like "today" and "tomorrow" should be interpreted based on the current date when processing the text.
- Specific dates can be mentioned in various formats, including full dates like "June 1, 2023", date strings like "2023-06-01", or abbreviated forms like "Jan 15, 22".
- The extracted timestamp should be the earliest mentioned timestamp in the text.
- In case multiple timestamps are mentioned, compare the timestamps and extract the oldest one.
- If no explicit timestamps are found, assume the current date as the oldest timestamp and return today's timestamp in 'YYYY-MM-DDT00:00:00' format.

Examples:
1. "Please give today's data for the temperature sensor in Verna ground floor." -> {datetime.now()}

2. "Give me a report for the humidity sensor on the first floor yesterday."->{(datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")}

3. "I need a report starting from June 1, 2023."-> '2023-06-01T00:00:00'

4. "I need a report from last month."->{(datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d")}

5. "Give me report from 1st Jan 2021 to 3rd March 2021." -> '2021-01-01T00:00:00'

6. "No specific dates mentioned in the text." -> {datetime.now()}

Given the following text:
        Question:{text} 
        Answer: 
""")
        to_time = llm(f"""
Instructions:
1. The purpose of this interaction is to extract the latest timestamp mentioned in the given text.
2. The extracted timestamp should be in the format of 'YYYY-MM-DDTHH:MM:SS', such as "today", "tomorrow", or specific dates.
3. The latest timestamp refers to the most recent datetime.
4. If no timestamps are found, the response should be Current's timestamp in 'YYYY-MM-DDTHH:MM:SS' format.
5. Ensure that the extracted timestamp is in a consistent and standardized format for further processing or analysis.
6. Latest timestamp should be the current timestamp by default unless specified.
7. If only one timestamp is specified, understand the text context and assume latest timestamp from it.
8. By default, the latest date will be the current timestamp unless specified in the text.

Elaborated Guidelines:
- The text may include sentences or phrases related to timestamps, such as "Give me the data from yesterday", "I need a report starting from June 1, 2023", or "What was the value on January 15, 2022?".
- Relative references like "today" and "tomorrow" should be interpreted based on the current date when processing the text.
- Specific dates can be mentioned in various formats, including full dates like "June 1, 2023", date strings like "2023-06-01", or abbreviated forms like "Jan 15, 22".
- The extracted timestamp should be the latest mentioned timestamp in the text.
- In case multiple timestamps are mentioned, compare the timestamps and extract the latest one.
- If no explicit timestamps are found, assume the current date as the latest timestamp and return today's timestamp in 'YYYY-MM-DDTHH:MM:SS' format.

Examples:
1. "Please give today's data for the temperature sensor in Verna ground floor." -> {datetime.now()}

2. "Give me a report for the humidity sensor on the first floor yesterday." -> "{(datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")}T23:59:59"

3. "I need a report starting from June 1, 2023." -> 2023-06-01T23:59:59

4. "I need a report from last month." -> {(datetime.now().replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")}T23:59:59

5. "Give me report from 1st Jan 2021 to 3rd March 2021." -> 2021-03-03T23:59:59
   
6. "No specific dates mentioned in the text." ->{datetime.now()}

Given the following text:
        Question:{text} 
        Answer: 

""")
        
        print(f"Fetching Sensor's data {sensor_id} between '{from_time.strip()}' to '{to_time.strip()}'")
        
        if isinstance(sensor_id, int):
            sensor_data : str = get_response(f'/genesis/data/report/interactive?sensor_id={sensor_id}&timestamp_from={from_time.strip()}Z&timestamp_to={to_time.strip()}Z')
            try:
                return sensor_data['data'][0]
            except TypeError:
                return "No Data"
        elif isinstance(sensor_id, str):
            return "Sensor Not Found"
        elif isinstance(sensor_id, list):
            return f"Found {len(sensor_id)}No. of sensor here is list {sensor_id}"
        

    """
1. The purpose of this interaction is to extract the minimum and maximum timestamps mentioned in the given text.
2. The extracted timestamps should be in the format of dates, such as "today", "tomorrow", or specific dates.
3. The minimum timestamp refers to the earlier date, and the maximum timestamp refers to the later date.
4. If no timestamps are found, the response should be "No timestamps found."
5. The text may contain references to "today" and "tomorrow" for relative timestamps.
6. Other specific dates can also be mentioned, such as "June 1, 2023" or "2023-06-01".
7. Ensure that the extracted timestamps are in a consistent and standardized format for further processing or analysis.

Examples:
- "Please complete the task by today." -> Minimum timestamp: today, Maximum timestamp: today
- "The meeting is scheduled for tomorrow." -> Minimum timestamp: tomorrow, Maximum timestamp: tomorrow
- "The event will take place on June 1, 2023." -> Minimum timestamp: June 1, 2023, Maximum timestamp: June 1, 2023
- "The project deadline is approaching. Ensure all tasks are completed by tomorrow." -> Minimum timestamp: today, Maximum timestamp: tomorrow
- "No specific dates mentioned in the text." -> No timestamps found.
"""

class Unit():
    def get_unit_id(self, location : Optional[str] = "") -> int:
        unit_id = self.fetch_unit_id( location)
        return unit_id
    
    def fetch_unit_id(self, location):
        response = get_response(f"/genesis/query/unit/find?unit_name={location}")
        if isinstance(response, list):
            if len(response)==1:
                return response[0]['unit_id']
            else:
                unit_list = []
                for sensor in response:
                    unit_list.append(sensor['unit_urn'])
                return unit_list
        else:
            return f"request failed with status code: {response}"        

    def fetch_unit_status(self,location : int):
        response = get_response(f"/genesis/query/unit_status?unit_id={location}")
        try:
            return response['unit_status']['unit_health']['code_name']
        except TypeError as e:
            return f"request failed with status code: {response}"

    def get_unit_status(self,text):
        location = llm(f"""
Instructions:
1. The purpose of this interaction is to extract a location mentioned in the given text.
2. The extracted location should be returned as a string in capital case.
3. The location can be in the format 'VER_W1_B2_GF_B' or 'VER_W1_B2_GF_A'.
4. The location may also contain aliases such as 'Verna Ground floor B' or 'Verna Ground floor A'.
5. If no location is found, the response should be 'No location found.'
6. If we don't know the warehouse from the location name, it can be represented by '%' (e.g., 'Verna building 2 Ground Floor' is 'VER_W%_B2_GF').
7. If we don't know the floor from the location name, it can be represented by '%' (e.g., 'Verna Warehouse building 2' is 'VER_W1_B2_%').
8. If we don't know the building from the location name, it can be represented by '%' (e.g., 'Verna Warehouse 2 Ground Floor' is 'VER_W2_%_GF').

Examples:
- 'VER_W1_B2_GF_B' corresponds to 'Verna Warehouse 1 Building 2 Ground Floor B'.
- 'VER_W2_B5_FF_C' corresponds to 'Verna Warehouse 2 Building 5 First Floor C'.
- 'KUD_W1_B2_GF_X_1' corresponds to 'Kundai Warehouse 1 Building 2 Ground Floor X 1'.
- 'VER_W1_WARLVL_WARLVL_WARLVL' corresponds to 'Verna Warehouse level'.
- 'KUD_W1_WARLVL_WARLVL_WARLVL' corresponds to 'Kundai Warehouse level'.
- 'VER_W%_B2_GF' corresponds to 'Verna building 2 Ground Floor'
- 'VER_W1_B2_%' corresponds to 'Verna Warehouse building 2'
- 'VER_W2_%_GF' corresponds to 'Verna Warehouse 2 Ground Floor'                                              
                       
                       


Given the following text:
Question: {text}
Answer: 
                    """)
        
        print(f"recovered info location {location}")


        unit_id : int = self.get_unit_id(location.strip())
        if isinstance(unit_id, int):
            unit_status : str = self.fetch_unit_status(unit_id)
            return unit_status
        elif isinstance(unit_id, str):
            return "Unit Not Found"
        elif isinstance(unit_id, list):
            return f"Found {len(unit_id)}No. of Unit here is list {unit_id}"
    
    def get_all_unit_list(self,text):
            response = get_response(f"/genesis/query/unit/list")
            if isinstance(response, list):
                unit_list = []
                for sensor in response:
                    unit_list.append(sensor['unit_urn'])
                return unit_list
            else:
                return f"request failed with status code: {response}" 

report_tool = Tool(
    name='get_report',
    func=Sensor().get_data_for_sensor,
    description="Use this tool to fetch data from a specific sensor. User has to specify the sensor type and location to identify sensor and fetch data report"

)

sensor_status_tool= Tool(
    name='get_sensor_status',
    func=Sensor().get_sensor_status,
    description='Use this tool to retrieve the status of a sensor based on its sensor type and location. The status can be one of the following values: "Active", "Inactive", "Normal", "Out of Range", or "Sensor Not Found". Provide the sensor type and location as input to get the corresponding status.'
    )

fetch_all_sensor_tool = Tool(
    name='get_all_sensor_info',
    func=Sensor().get_all_sensor_list,
    description="Use this tool to retrieve a list of all available sensors. No specific location or sensor type is required as input. Simply call this tool without any arguments when user asks for the list of all sensors. if any information about sensor type or location is present try sensor status method "
)

unit_status_tool = Tool(
    name="get_unit_status",
    func=Unit().get_unit_status,
    description="Use this tool when user asks for list of all unit if found send text input else send none"
)

fetch_all_unit_tool = Tool(
    name='get_all_unit_info',
    func=Unit().get_all_unit_list,
    # description="Use this tool when user asks for list of all unit if found send text input else send none"
    description='Use this tool to retrieve a comprehensive list of all available units. If the units are found, the response will contain the text input with the list of units. If no units are found, the response will be None. Please note that the list of units may vary depending on the system configuration and available data.'
)