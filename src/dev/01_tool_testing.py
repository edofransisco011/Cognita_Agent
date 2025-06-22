import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults

def test_tavily_tool():
    """
    A script to test the Tavily Search API and its LangChain Tool wrapper.
    """
    print("--- Starting Tool Test ---")

    # 1. Load environment variables from .env file
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in .env file. Please add it.")

    # 2. Test the TavilyClient directly
    print("\n--- Testing TavilyClient directly ---")
    client = TavilyClient(api_key=tavily_api_key)
    response = client.search(query="What are the latest advancements in Large Language Models as of 2025?")
    print("Direct API Response Keys:", response.keys())
    # Print content of the first result
    print("Content of first result:", response['results'][0]['content'])

    # 3. Test the LangChain TavilySearchResults Tool
    print("\n--- Testing LangChain TavilySearchResults tool ---")
    tavily_tool = TavilySearchResults(max_results=5)

    # Inspect the tool's properties
    print("Tool Name:", tavily_tool.name)
    print("Tool Description:", tavily_tool.description)

    # Invoke the tool
    tool_results = tavily_tool.invoke({"query": "What are the latest advancements in Large Language Models as of 2025?"})

    print("Tool Invocation Results (type):", type(tool_results))
    print("First few characters of tool results:", tool_results[:200])

    print("\n--- Tool Test Finished ---")

if __name__ == "__main__":
    test_tavily_tool()