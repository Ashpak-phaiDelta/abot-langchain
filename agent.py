from langchain.agents import initialize_agent , load_tools, AgentType

# from math_agent import zero_shot_agent
from llm import llm
from Tools import sensor_status


# print(zero_shot_agent("what is 10 +19"))

tools = [
    sensor_status.sensor_status_tool, 
    sensor_status.fetch_all_sensor_tool, 
    sensor_status.unit_status_tool, 
    sensor_status.fetch_all_unit_tool,
    sensor_status.report_tool
    ]
# tools =[sensor_status.sensor_info_tool]



text_agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations =3
)


# print(text_agent("give me name of all sensors "))

# print(text_agent("VER_W1_B2_FF_A_1_temp"))
# print(text_agent("i want Temperature sensor status for verna first floor a"))
# print(text_agent("I need Status Temperature sensor in Verna Ground floor B "))
# print(text_agent("Give me list of all Temperature sensor"))
# print(text_agent("what is current status for unit VER_W1_B2_GF_A"))
# print(text_agent("Ground floor B temperature sensor stats"))

# print(text_agent("Give me report for temperature sensor in verna warehouse 1 building 2 first floor a for after May"))
print(text_agent.run(input("Enter: ")))
# text_agent.run("Enter")