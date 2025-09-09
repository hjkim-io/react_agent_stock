from typing import Dict, Any
from langchain_core.messages import SystemMessage
from src.states.types import State
from src.agents.llm import get_llm_by_type
from src.prompts.template import get_system_prompt_from_yaml
from src.tools.stock_tools import StockToolkit
from langgraph.prebuilt import create_react_agent
import datetime

# stock_tools만 사용
all_tools = StockToolkit().get_tools()

def create_limited_messages(state: State) -> list:
    system_message = SystemMessage(
        content=get_system_prompt_from_yaml("stock_price_agent") + 
        f"\n오늘 날짜는: {datetime.datetime.now().strftime('%Y%m%d')}입니다."
    )
    
    return [system_message] + state["messages"]

stock_agent = create_react_agent(
    get_llm_by_type("basic"),
    tools=all_tools,
    prompt=create_limited_messages,
)

async def main_agent_node(state: State) -> Dict[str, Any]:
    result = await stock_agent.ainvoke(state)
    return {"messages": [result["messages"][-1]]}
