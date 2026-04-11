import datetime
from livekit.agents import RunContext, function_tool

class TimeTools:
    @function_tool
    async def get_time(self, context: RunContext):
        """Gets the current system time to tell the user and update the frontend display."""
        print("\n\033[92m[TOOL EXECUTION]\033[0m 🛠️ The LLM is calling the 'get_time' tool...")
        
        now = datetime.datetime.now()
        readable_time = now.strftime("%I:%M %p")
        full_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        return f"The current system time is {readable_time}."
