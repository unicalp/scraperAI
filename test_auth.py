import os
import sys

print("Python version:", sys.version)  # Add this line
print("Current working directory:", os.getcwd()) # Add this line
print("GOOGLE_APPLICATION_CREDENTIALS:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

# The rest of your code starts here...
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from browser_use import Agent

load_dotenv()

async def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set in the .env file.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key=gemini_api_key)
    agent = Agent(
        task="search google and find out step by step of getting aws activate",
        llm=llm,
    )
    await agent.run()

asyncio.run(main())