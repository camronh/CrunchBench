
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openrouter = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
    model="deepseek/deepseek-chat",
)
