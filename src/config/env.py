import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# LANGSMIT
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT","")

# MCP 설정
MCP_CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", "mcp_config.json")