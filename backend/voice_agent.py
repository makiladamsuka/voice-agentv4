from dotenv import load_dotenv
from livekit import agents, rtc
from tools import TimeTools, SearchTools
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai, deepgram, silero
from amplitude_tts import AmplitudeTTS, _drain_to_zero
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

class SimpleVoiceAgent(Agent, TimeTools, SearchTools):
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
        tts=AmplitudeTTS(model="aura-luna-en"),
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
    
    # --- Sentiment / Vader emotion analysis ---
    import socket
    import json
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    analyzer = SentimentIntensityAnalyzer()

    def analyze_and_send_emotion(text: str):
        if not text or not text.strip(): return
        if len(text.split()) < 2: return
        
        scores = analyzer.polarity_scores(text)
        comp = scores['compound']
        
        if comp > 0.6: emotion = "happy"
        elif comp > 0.2: emotion = "warm"
        elif comp < -0.6: emotion = "angry"
        elif comp < -0.2: emotion = "sad"
        else: emotion = "engaged"

        try:
            payload = json.dumps({"command": "emotion", "emotion": emotion}).encode("utf-8")
            sock.sendto(payload, ("127.0.0.1", 9000))
            print(f"🤖 [Vader] '{text[:40]}...' -> Score: {comp:.2f} -> {emotion}")
        except:
            pass

    @session.on("agent_speech_committed")
    def on_agent_speech_committed(msg):
        try:
            text = str(msg)
            if hasattr(msg, "content"): text = msg.content
            elif hasattr(msg, "text"): text = msg.text
            elif isinstance(msg, dict) and "content" in msg: text = msg["content"]
            analyze_and_send_emotion(text)
        except Exception as e:
            print("Vader Error:", e)

    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        try:
            text = str(msg)
            if hasattr(msg, "content"): text = msg.content
            elif hasattr(msg, "text"): text = msg.text
            elif isinstance(msg, dict) and "content" in msg: text = msg["content"]
            analyze_and_send_emotion(text)
        except Exception as e:
            pass

    # Drain amplitude to zero when agent finishes talking (insurance)
    @ctx.room.on("active_speakers_changed")
    def on_speakers_changed(speakers):
        speaking = any(
            p.sid == ctx.room.local_participant.sid for p in speakers
        )
        if not speaking:
            _drain_to_zero()

    # Keeps session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="campus-greeting-agent"))
