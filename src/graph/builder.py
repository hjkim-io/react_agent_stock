from langgraph.graph import StateGraph, END
from src.states.types import State
from src.agents.main_agent import main_agent_node

def build_hybrid_graph():
    builder = StateGraph(State)
    
    # 메인 에이전트 노드 추가
    builder.add_node("main_agent", main_agent_node)
    
    # 진입점 설정
    builder.set_entry_point("main_agent")
    
    # 메인 에이전트 노드에서 바로 종료
    builder.add_edge("main_agent", END)
    
    return builder
