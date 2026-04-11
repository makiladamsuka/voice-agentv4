from livekit.agents import RunContext, function_tool
from duckduckgo_search import DDGS
import asyncio
import logging

class SearchTools:
    @function_tool(description="Searches the web for current events or missing knowledge.")
    async def search_web(self, context: RunContext, query: str):
        """Searches the internet for real-time information about a given query."""
        print(f"\n\033[96m[TOOL EXECUTION]\033[0m 🌐 The LLM is calling the 'search_web' tool for: '{query}'...")
        
        if not query:
            return "No query provided."

        try:
            results = await asyncio.to_thread(lambda: DDGS().text(query, max_results=3))
            
            if not results:
                return "No search results found."
            
            formatted = []
            for r in results:
                formatted.append(f"{r['title']}: {r['body']}")
            
            answer = " | ".join(formatted)
            # Keep it short for TTS
            answer = answer[:500]
            print(f"\033[92m[SEARCH RESULT]\033[0m {answer}")
            return f"Web search results: {answer}"
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return f"Search failed: {e}"
