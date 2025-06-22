import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch # New recommended import

def test_tavily_tool():
    """
    A script to test the LATEST Tavily Search tool.
    """
    print("--- Starting Tool Test (v2) ---")

    # 1. Load environment variables
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in .env file. Please add it.")

    # 2. Instantiate the LangChain TavilySearch tool
    # This is the new, recommended way.
    print("\n--- Testing LangChain TavilySearch tool ---")
    tavily_tool = TavilySearch(max_results=5)

    # 3. Inspect the tool's properties
    print("Tool Name:", tavily_tool.name)
    print("Tool Description:", tavily_tool.description)
    
    # 4. Invoke the tool
    # Note: The output format of this new tool is a list of strings by default,
    # which is often cleaner for an LLM to process.
    tool_results = tavily_tool.invoke("What are the latest advancements in Large Language Models as of 2025?")
    
    print("\nTool Invocation Results (type):", type(tool_results))
    print("\nFirst Result:")
    print(tool_results[0]) # Print the first result string
    
    print("\n--- Tool Test Finished ---")

if __name__ == "__main__":
    test_tavily_tool()