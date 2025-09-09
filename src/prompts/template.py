import os
import yaml
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.agents.llm import get_llm_by_type

def load_prompt_config(prompt_name: str) -> Dict[str, Any]:
    """지정된 이름의 .yaml 프롬프트 설정 파일을 로드하고 파싱합니다."""
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, f"{prompt_name}.yaml")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt config file not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Failed to load or parse prompt config: {prompt_name}")
        
    return config

def create_runnable_from_prompt(prompt_name: str) -> Runnable:
    """YAML 설정 파일로부터 프롬프트, LLM, 출력 파서를 포함한 Runnable 체인을 생성합니다."""
    config = load_prompt_config(prompt_name)
    
    prompt_template_str = config.get("prompt")
    if not prompt_template_str:
        raise ValueError(f"'prompt' key not found in {prompt_name}.yaml")
    
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
    
    model_type = config.get("model", "basic")
    llm = get_llm_by_type(model_type)
    
    response_format = config.get("response_format")
    if response_format and response_format.get("type") == "json_schema":
        json_schema_config = response_format.get("json_schema", {})
        schema = json_schema_config.get("schema")
        
        if schema:
            schema_title = json_schema_config.get("name")
            schema_description = json_schema_config.get("description")
            
            if schema_title:
                schema["title"] = schema_title
            if schema_description:
                schema["description"] = schema_description
            
            llm = llm.with_structured_output(schema)

    chain = prompt_template | llm
    return chain

def format_messages_for_prompt(messages: List[BaseMessage]) -> str:
    """메시지 리스트를 프롬프트용 문자열로 변환합니다."""
    dialogue = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            dialogue.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            dialogue.append(f"AI: {msg.content}")
    return "\n".join(dialogue)

def get_system_prompt_from_yaml(prompt_name: str) -> str:
    """YAML 설정 파일에서 프롬프트 문자열만 로드합니다."""
    config = load_prompt_config(prompt_name)
    prompt_template_str = config.get("prompt")
    if not prompt_template_str:
        raise ValueError(f"'prompt' key not found in {prompt_name}.yaml")
    return prompt_template_str
