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

# ── State 1: Mood Tracking (VADER) ──────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()

def _send_vader_emotion(text: str, is_agent: bool = False):
    """Layer 1: VADER — slow mood backdrop from utterance sentiment."""
    if not text or len(text.split()) < 2:
        return
    words = text.split()
    word_count = len(words)
    comp = _analyzer.polarity_scores(text)["compound"]
    
    emotion = "engaged" # default floor
    
    if comp > 0.6: 
        emotion = "happy"
    elif comp > 0.2: 
        emotion = "warm"
    elif comp < -0.2: 
        if is_agent or "sorry" in text.lower():
            emotion = "apologetic"
        else:
            emotion = "sad"
    elif comp < -0.6:
        emotion = "angry"
    
    # Special context overrides
    if -0.2 <= comp <= 0.2 and word_count > 10:
        emotion = "engaged"
    if comp > 0.3 and word_count > 15 and is_agent:
        emotion = "proud"

    _udp({"command": "emotion", "emotion": emotion})
    print(f"🤖 [Vader L1] {'Agent' if is_agent else 'User'} said: '{text[:30]}...' -> {comp:.2f} -> {emotion}")

# ── State 2: Conversation state machine ─────────────────────────────────────────────────
_thinking_task: asyncio.Task | None = None
_awkward_timer_task: asyncio.Task | None = None
_smart_wait_task: asyncio.Task | None = None

async def _thinking_cycle(word_count: int):
    """
    Stage 2 Processing:
    0 - 0.5s: nodding
    0.5s - 2.0s: thinking OR concentrating
    2.0s+: remembering
    """
    # Acknowledge immediately
    _set_conv_state("nodding", "nodding")
    await asyncio.sleep(0.5)

    # Decide between thinking and concentrating
    base_state = "concentrating" if word_count > 15 else "thinking"
    _set_conv_state(base_state, base_state)
    
    await asyncio.sleep(1.5) # 0.5 + 1.5 = 2.0s mark
    
    # Transition to deep memory
    _set_conv_state("remembering", "remembering")
    print(f"🧠 [ConvState L2] Transitioned to REMEMBERING...")
    
    while True: # Keep cycling subtle variations if LLM is slow
        await asyncio.sleep(3.0)
        _set_conv_state("thinking", "thinking")
        await asyncio.sleep(3.0)
        _set_conv_state("remembering", "remembering")

async def _awkward_timer():
    """Timer that triggers 'awkward' after 5s of silence."""
    await asyncio.sleep(5.0)
    _udp({"command": "conv_state", "state": "waiting", "emotion": "awkward"})
    print("😶 [ConvState L2] Situation is getting AWKWARD...")

def _set_conv_state(state: str, emotion: str | None = None):
    """Send a conversation state packet. Cancels active timers."""
    global _thinking_task, _awkward_timer_task, _smart_wait_task
    if _thinking_task and not _thinking_task.done():
        _thinking_task.cancel()
    if _awkward_timer_task and not _awkward_timer_task.done():
        _awkward_timer_task.cancel()
    if _smart_wait_task and not _smart_wait_task.done():
        _smart_wait_task.cancel()
    
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
    global _thinking_task, _awkward_timer_task, _smart_wait_task

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

    # ── State 1: LISTENING / WAITING (Precise VAD) ───────────────────────────
    async def _hearing_reflex():
        """Stage 1: The 'Startle' — robot perks up when hearing voice."""
        # Pulse to excited immediately to show hearing
        _udp({"command": "conv_state", "state": "listening", "emotion": "excited"})
        await asyncio.sleep(0.4)
        # Settle into attentive listening
        _udp({"command": "conv_state", "state": "listening", "emotion": "attentive"})

    @session.on("user_started_speaking")
    def on_user_started():
        asyncio.create_task(_hearing_reflex())

    @session.on("user_stopped_speaking")
    def on_user_stopped():
        global _awkward_timer_task, _smart_wait_task
        # Instead of snapping to waiting immediately, we wait slightly
        # to see if 'user_speech_committed' is about to fire.
        async def _smart_wait():
            await asyncio.sleep(1.2) # Grace period for STT to commit
            if ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                # If we aren't thinking/speaking yet, go back to waiting
                _set_conv_state("waiting", "attentive")
        
        # Start awkward countdown
        if _awkward_timer_task and not _awkward_timer_task.done():
            _awkward_timer_task.cancel()
        _awkward_timer_task = asyncio.create_task(_awkward_timer())
        
        # Clean up existing smart wait and start a new one
        if _smart_wait_task and not _smart_wait_task.done():
            _smart_wait_task.cancel()
        _smart_wait_task = asyncio.create_task(_smart_wait())

    # ── State 2: THINKING — user finished, LLM is processing ──────────────────
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        global _thinking_task
        text = str(msg)
        if hasattr(msg, "content"):  text = msg.content
        elif hasattr(msg, "text"):   text = msg.text
        
        # CLEANUP: Filter out junk for word count
        junk = ["uh", "um", "ah", "er", "hmm", "okay", "so", "well"]
        clean_words = [w for w in text.lower().split() if w not in junk]
        word_count = len(clean_words)
        
        _thinking_task = asyncio.create_task(_thinking_cycle(word_count))
        
        try:
            _send_vader_emotion(text, is_agent=False)
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
            _send_vader_emotion(text, is_agent=True)
        except Exception as e:
            print("Vader Error:", e)

    # ── State 4: AGENT SILENCE Tracking ───────────────────────────────────────
    @session.on("agent_started_speaking")
    def on_agent_started():
        _set_conv_state("speaking", "engaged")

    @session.on("agent_stopped_speaking")
    def on_agent_stopped():
        global _awkward_timer_task
        _drain_to_zero()
        _set_conv_state("waiting", "attentive")
        if _awkward_timer_task and not _awkward_timer_task.done():
            _awkward_timer_task.cancel()
        _awkward_timer_task = asyncio.create_task(_awkward_timer())


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
    global _thinking_task, _awkward_timer_task

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

    # ── State 1: LISTENING / WAITING (Precise VAD) ───────────────────────────
    async def _hearing_reflex():
        """Stage 1: The 'Startle' — robot perks up when hearing voice."""
        # Pulse to excited immediately to show hearing
        _udp({"command": "conv_state", "state": "listening", "emotion": "excited"})
        await asyncio.sleep(0.4)
        # Settle into attentive listening
        _udp({"command": "conv_state", "state": "listening", "emotion": "attentive"})

    @session.on("user_started_speaking")
    def on_user_started():
        asyncio.create_task(_hearing_reflex())

    @session.on("user_stopped_speaking")
    def on_user_stopped():
        global _awkward_timer_task
        # Instead of snapping to waiting immediately, we wait slightly
        # to see if 'user_speech_committed' is about to fire.
        async def _smart_wait():
            await asyncio.sleep(1.2) # Grace period for STT to commit
            if ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                # If we aren't thinking/speaking yet, go back to waiting
                _set_conv_state("waiting", "attentive")
        
        # Start awkward countdown
        if _awkward_timer_task and not _awkward_timer_task.done():
            _awkward_timer_task.cancel()
        _awkward_timer_task = asyncio.create_task(_awkward_timer())
        asyncio.create_task(_smart_wait())

    # ── State 2: THINKING — user finished, LLM is processing ──────────────────
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        global _thinking_task
        text = str(msg)
        if hasattr(msg, "content"):  text = msg.content
        elif hasattr(msg, "text"):   text = msg.text
        
        # CLEANUP: Filter out junk for word count
        junk = ["uh", "um", "ah", "er", "hmm", "okay", "so", "well"]
        clean_words = [w for w in text.lower().split() if w not in junk]
        word_count = len(clean_words)
        
        _thinking_task = asyncio.create_task(_thinking_cycle(word_count))
        
        try:
            _send_vader_emotion(text, is_agent=False)
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
            _send_vader_emotion(text, is_agent=True)
        except Exception as e:
            print("Vader Error:", e)

    # ── State 4: AGENT SILENCE Tracking ───────────────────────────────────────
    @session.on("agent_started_speaking")
    def on_agent_started():
        global _awkward_timer_task, _thinking_task
        # Hard cancel any thinking/waiting tasks once audio is live
        if _thinking_task and not _thinking_task.done():
            _thinking_task.cancel()
        if _awkward_timer_task and not _awkward_timer_task.done():
            _awkward_timer_task.cancel()
        _set_conv_state("speaking", "engaged")

    @session.on("agent_stopped_speaking")
    def on_agent_stopped():
        global _awkward_timer_task
        _drain_to_zero()
        _set_conv_state("waiting", "attentive")
        if _awkward_timer_task and not _awkward_timer_task.done():
            _awkward_timer_task.cancel()
        _awkward_timer_task = asyncio.create_task(_awkward_timer())

    # Keeps session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="campus-greeting-agent"))
