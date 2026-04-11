from dotenv import load_dotenv
from livekit import agents, rtc
from tools import TimeTools, SearchTools, EmotionTools
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai, deepgram, silero
import os
import asyncio
import datetime
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

def prewarm(proc: agents.JobProcess):
    """Pre-load models to cache them in the worker process before jobs arrive."""
    silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.2, 
        prefix_padding_duration=0.2
    )

class SimpleVoiceAgent(Agent, TimeTools, SearchTools, EmotionTools):
    def __init__(self):
        from prompt import SYSTEM_INSTRUCTIONS
        super().__init__(
            instructions=SYSTEM_INSTRUCTIONS
        )

async def entrypoint(ctx: agents.JobContext):
    # Create session immediately 
    session = AgentSession(
        turn_handling=agents.TurnHandlingOptions(
            interruption={"mode": "vad"}
        ),
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-luna-en"),
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.2, 
            prefix_padding_duration=0.2
        ),
        llm=openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        ),
    )
    
    # Create agent
    agent = SimpleVoiceAgent()
    
    # START SESSION
    print("🚀 Starting LiveKit session...")
    await session.start(room=ctx.room, agent=agent)
    
    await session.say("Hello! I am your voice assistant. How can I help you today?")
    
    # --- UDP Pulse Bridge ---
    import socket
    import json
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send_pulse(power: float):
        try:
            payload = json.dumps({"speak_pulse": power}).encode("utf-8")
            sock.sendto(payload, ("127.0.0.1", 9000))
        except:
            pass

    # Bind to every possible LiveKit speaking event name to ensure it fires
    # regardless of which SDK version you are running!
    @session.on("agent_speech_started")
    def on_speech_started(*args, **kwargs):
        send_pulse(1.0)
        
    @session.on("agent_started_speaking")
    def on_started_speaking(*args, **kwargs):
        send_pulse(1.0)

    @session.on("agent_speech_stopped")
    def on_speech_stopped(*args, **kwargs):
        send_pulse(0.0)
        
    @session.on("agent_stopped_speaking")
    def on_stopped_speaking(*args, **kwargs):
        send_pulse(0.0)
        
    # Keeps session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="campus-greeting-agent"))
