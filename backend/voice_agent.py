from dotenv import load_dotenv
from livekit import agents, rtc
from tools import TimeTools, SearchTools
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
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    analyzer = SentimentIntensityAnalyzer()
    
    def send_pulse(power: float):
        try:
            payload = json.dumps({"speak_pulse": power}).encode("utf-8")
            sock.sendto(payload, ("127.0.0.1", 9000))
        except:
            pass

    def analyze_and_send_emotion(text: str):
        if not text or not text.strip(): return
        
        # We don't want to analyze single filler words
        if len(text.split()) < 2: return
        
        scores = analyzer.polarity_scores(text)
        comp = scores['compound']
        
        emotion = "neutral"
        if comp > 0.6: emotion = "happy"
        elif comp > 0.2: emotion = "warm"
        elif comp < -0.6: emotion = "angry"
        elif comp < -0.2: emotion = "sad"
        else: emotion = "engaged" # default thinking state while talking

        try:
            payload = json.dumps({"command": "emotion", "emotion": emotion}).encode("utf-8")
            sock.sendto(payload, ("127.0.0.1", 9000))
            print(f"🤖 [Vader] Evaluated Agent Thought: '{text[:40]}...' -> Score: {comp:.2f} -> Eye Emotion: {emotion}")
        except:
            pass

    # Try to hook the generated text before it's spoken
    @session.on("agent_speech_committed")
    def on_agent_speech_committed(msg):
        try:
            # Different SDK versions wrap this differently
            text = str(msg)
            if hasattr(msg, "content"): text = msg.content
            elif hasattr(msg, "text"): text = msg.text
            elif isinstance(msg, dict) and "content" in msg: text = msg["content"]
            analyze_and_send_emotion(text)
        except Exception as e:
            print("Vader Error:", e)

    # Note: We also evaluate what the USER says to pre-react with emotion!
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

    # Hook into raw WebRTC room events to detect when the local agent is outputting audio
    @ctx.room.on("active_speakers_changed")
    def on_speakers_changed(speakers):
        # Check if the agent (local participant) is currently producing audio
        speaking = False
        for p in speakers:
            if p.sid == ctx.room.local_participant.sid:
                speaking = True
                break
                
        if speaking:
            print("🔊 [Audio Matrix] Agent is actively speaking -> BOUNCE !")
            send_pulse(1.0)
        else:
            print("🔇 [Audio Matrix] Agent is silent -> STOP BOUNCE")
            send_pulse(0.0)
        
    # Keeps session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="campus-greeting-agent"))
