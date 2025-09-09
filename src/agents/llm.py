from langchain_openai import ChatOpenAI
from typing import Optional
from src.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
from typing import Literal
LLMType = Literal["basic", "reasoning"]

AGENT_LLM_MAP: dict[str, LLMType] = {
    "supervisor": "basic",
    "billing": "basic",
    "chitchat": "basic",
    "basic": "basic"
}

def create_openai_llm(
    model: str = OPENAI_MODEL,
    base_url: Optional[str] = OPENAI_BASE_URL,
    api_key: str = OPENAI_API_KEY,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}
    
    if base_url:
        llm_kwargs["base_url"] = base_url
    elif api_key:
        llm_kwargs["api_key"] = api_key
        
    return ChatOpenAI(**llm_kwargs)

_llm_cache: dict[LLMType, ChatOpenAI] = {}

def get_llm_by_type(llm_type: LLMType) -> ChatOpenAI:
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]
    
    # if llm_type == "basic":
    llm = create_openai_llm()
    # elif llm_type == "reasoning":
        # llm = create_openai_llm(temperature=0.1) # tbd -> langmanus 에서착안 (나중에 비즈니스 상 모델 바꾸거나 할때 사용)
    # else:
        # raise ValueError(f"Unknown LLM type: {llm_type}")
    
    _llm_cache[llm_type] = llm
    return llm