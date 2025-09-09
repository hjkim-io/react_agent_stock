from typing import Literal, Any, Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

class State(MessagesState):
    """전체 워크플로우의 상태를 관리하는 기본 클래스"""
    next_agent: str
    # 관련 컨텍스트
    context: Dict[str, Any]