from dotenv import load_dotenv
from livekit import agents, rtc
from tools import TimeTools, SearchTools
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai, deepgram, silero
from amplitude_tts import AmplitudeTTS, _drain_to_zero
import os
import asyncio
import datetime
import socket
import json
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# ── Shared UDP socket ──────────────────────────────────────────────────────────
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_EYES = ("127.0.0.1", 9000)

def _udp(payload: dict):
    try:
        _sock.sendto(json.dumps(payload).encode(), _EYES)
    except Exception:
        pass

# ── VADER analyzer ─────────────────────────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()

def _send_vader_emotion(text: str):
    """Layer 1: VADER — slow mood backdrop from utterance sentiment."""
    if not text or len(text.split()) < 2:
        return
    comp = _analyzer.polarity_scores(text)["compound"]
    if comp > 0.6:    emotion = "happy"
    elif comp > 0.2:  emotion = "warm"
    elif comp < -0.6: emotion = "angry"
    elif comp < -0.2: emotion = "sad"
    else:             emotion = "engaged"
    _udp({"command": "emotion", "emotion": emotion})
    print(f"🤖 [Vader L1] '{text[:40]}' -> {comp:.2f} -> {emotion}")

# ── Conversation state machine ─────────────────────────────────────────────────
_thinking_task: asyncio.Task | None = None

async def _thinking_cycle():
    """Cycle through thinking → remembering → concentrating while LLM processes."""
    emotions = ["thinking", "remembering", "concentrating"]
    i = 0
    while True:
        e = emotions[i % len(emotions)]
        _udp({"command": "conv_state", "state": "thinking", "emotion": e})
        print(f"🧠 [ConvState L2] THINKING cycle -> {e}")
        i += 1
        await asyncio.sleep(1.5)

def _set_conv_state(state: str, emotion: str | None = None):
    """Send a conversation state packet. Cancels any running thinking cycle."""
    global _thinking_task
    if _thinking_task and not _thinking_task.done():
        _thinking_task.cancel()
        _thinking_task = None
    _udp({"command": "conv_state", "state": state, "emotion": emotion or state})
    print(f"👁  [ConvState L2] -> {state} ({emotion or state})")


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
        super().__init__(instructions=SYSTEM_INSTRUCTIONS)

async def entrypoint(ctx: agents.JobContext):
    global _thinking_task

    session = AgentSession(
        turn_handling=agents.TurnHandlingOptions(
            interruption={"mode": "vad"}
        ),
        stt=deepgram.STT(model="nova-2"),
        tts=AmplitudeTTS(model="aura-2-luna-en"),
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

    agent = SimpleVoiceAgent()
    print("🚀 Starting LiveKit session...")
    await session.start(room=ctx.room, agent=agent)
    await session.say("Oh hi! I am so happy you are talking to me!")

    # ── State 1: LISTENING — user starts speaking ──────────────────────────────
    try:
        @session.on("user_input_speech_started")
        def on_user_started(*args, **kwargs):
            _set_conv_state("listening", "attentive")
    except Exception:
        pass  # Not available in this SDK version

    # ── State 2: THINKING — user finished, LLM is processing ──────────────────
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        global _thinking_task
        _thinking_task = asyncio.create_task(_thinking_cycle())
        try:
            text = str(msg)
            if hasattr(msg, "content"):  text = msg.content
            elif hasattr(msg, "text"):   text = msg.text
            _send_vader_emotion(text)
        except Exception:
            pass

    # ── State 3: SPEAKING — agent response committed, audio starting ───────────
    @session.on("agent_speech_committed")
    def on_agent_speech_committed(msg):
        _set_conv_state("speaking", "engaged")
        try:
            text = str(msg)
            if hasattr(msg, "content"):  text = msg.content
            elif hasattr(msg, "text"):   text = msg.text
            _send_vader_emotion(text)
        except Exception as e:
            print("Vader Error:", e)

    # ── State 4: WAITING — agent finishes speaking, room goes quiet ───────────
    @ctx.room.on("active_speakers_changed")
    def on_speakers_changed(speakers):
        agent_speaking = any(
            p.sid == ctx.room.local_participant.sid for p in speakers
        )
        if not agent_speaking:
            _drain_to_zero()
            _set_conv_state("waiting", "attentive")

    # Keeps session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="campus-greeting-agent"))
