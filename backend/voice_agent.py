from dotenv import load_dotenv
from livekit import agents, rtc
from tools import TimeTools, SearchTools, ContentTools
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai, deepgram, silero
from amplitude_tts import AmplitudeTTS, _drain_to_zero
from text_filters import filter_leaked_tool_syntax
from image_server import ImageServer
from image_manager import ImageManager
from map_navigation import MapNavigator
from event_database import build_event_database
from greetings import generate_presence_greeting
import os
import asyncio
import socket
import json
import queue
import threading
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# ── Shared UDP socket (voice → robot eyes) ───────────────────────────────────
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_EYES = ("127.0.0.1", 9000)

# ── Presence arrivals (robot eyes → voice) ───────────────────────────────────
_presence_queue: queue.Queue[bool] = queue.Queue()
_VOICE_LISTENER_PORT = 9001

_global_image_server: ImageServer | None = None
_global_event_db = None


def _udp(payload: dict):
    try:
        _sock.sendto(json.dumps(payload).encode(), _EYES)
    except Exception:
        pass


def _voice_udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", _VOICE_LISTENER_PORT))
    except OSError as e:
        if e.errno in (98, 48):  # Linux / macOS "address already in use"
            print(
                f"Voice presence listener: UDP {_VOICE_LISTENER_PORT} already in use "
                "(stale voice_agent? stop old start_robot.py first)"
            )
            return
        raise
    print(f"Voice presence listener on UDP {_VOICE_LISTENER_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            msg = json.loads(data.decode("utf-8"))
            if msg.get("command") == "presence_arrival":
                _presence_queue.put(True)
        except Exception:
            pass


def _init_image_server():
    global _global_image_server
    if _global_image_server is None:
        assets_dir = Path(__file__).parent / "assets"
        _global_image_server = ImageServer(assets_dir, port=8080)
        _global_image_server.start()


def _build_event_db_sync():
    global _global_event_db
    try:
        assets_dir = Path(__file__).parent / "assets"
        _global_event_db = build_event_database(assets_dir)
    except Exception as e:
        print(f"Event database build failed: {e}")
        _global_event_db = None
    return _global_event_db


# ── State 1: Mood Tracking (VADER) ──────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()


def _send_vader_emotion(text: str, is_agent: bool = False):
    if not text or len(text.split()) < 2:
        return
    words = text.split()
    word_count = len(words)
    comp = _analyzer.polarity_scores(text)["compound"]

    emotion = "engaged"

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

    if -0.2 <= comp <= 0.2 and word_count > 10:
        emotion = "engaged"
    if comp > 0.3 and word_count > 15 and is_agent:
        emotion = "proud"

    _udp({"command": "emotion", "emotion": emotion})
    print(
        f"[Vader L1] {'Agent' if is_agent else 'User'} said: '{text[:30]}...' -> {comp:.2f} -> {emotion}"
    )


# ── State 2: Conversation state machine ─────────────────────────────────────
_thinking_task: asyncio.Task | None = None
_awkward_timer_task: asyncio.Task | None = None
_smart_wait_task: asyncio.Task | None = None


async def _thinking_cycle(word_count: int):
    _set_conv_state("nodding", "nodding")
    await asyncio.sleep(0.5)

    base_state = "concentrating" if word_count > 15 else "thinking"
    _set_conv_state(base_state, base_state)

    await asyncio.sleep(1.5)

    _set_conv_state("remembering", "remembering")
    print("[ConvState L2] Transitioned to REMEMBERING...")

    while True:
        await asyncio.sleep(3.0)
        _set_conv_state("thinking", "thinking")
        await asyncio.sleep(3.0)
        _set_conv_state("remembering", "remembering")


_session_live = False


async def _awkward_timer():
    await asyncio.sleep(5.0)
    if not _session_live:
        return
    _udp({"command": "conv_state", "state": "waiting", "emotion": "cheerful"})
    print("[ConvState L2] Long pause -> waiting (cheerful)")


def _set_conv_state(state: str, emotion: str | None = None):
    global _thinking_task, _awkward_timer_task, _smart_wait_task
    if _thinking_task and not _thinking_task.done():
        _thinking_task.cancel()
    if _awkward_timer_task and not _awkward_timer_task.done():
        _awkward_timer_task.cancel()
    if _smart_wait_task and not _smart_wait_task.done():
        _smart_wait_task.cancel()

    _udp({"command": "conv_state", "state": state, "emotion": emotion or state})
    print(f"[ConvState L2] -> {state} ({emotion or state})")


class CampusAgent(Agent, TimeTools, SearchTools):
    def __init__(self, image_server: ImageServer | None, event_db=None):
        from prompt import SYSTEM_INSTRUCTIONS

        assets_dir = Path(__file__).parent / "assets"
        self.image_manager = ImageManager(assets_dir)
        self.image_server = image_server
        self.event_db = event_db
        self.map_navigator = MapNavigator()
        self._room: rtc.Room | None = None
        self.content_tools = ContentTools(
            image_manager=self.image_manager,
            image_server=self.image_server,
            room_provider=lambda: self._room,
            map_navigator=self.map_navigator,
        )
        super().__init__(instructions=SYSTEM_INSTRUCTIONS)

    @function_tool
    async def list_available_events(
        self, filter_type: str = "all", context: RunContext = None
    ) -> str:
        """Lists all available events on campus."""
        return await self.content_tools.list_available_events(context)

    @function_tool
    async def show_event_poster(self, event_description: str, context: RunContext) -> str:
        """Displays an event poster on the frontend."""
        return await self.content_tools.show_event_poster(event_description, context)

    @function_tool
    async def show_competition_poster(
        self, competition_description: str, context: RunContext
    ) -> str:
        """Displays a competition poster on the frontend."""
        return await self.content_tools.show_competition_poster(
            competition_description, context
        )

    @function_tool
    async def show_campus_post(self, post_description: str, context: RunContext) -> str:
        """Displays a campus announcement poster on the frontend."""
        return await self.content_tools.show_campus_post(post_description, context)

    @function_tool
    async def show_location_map(self, location_query: str, context: RunContext) -> str:
        """Displays a campus location map on the frontend."""
        return await self.content_tools.show_location_map(location_query, context)

    @function_tool
    async def get_campus_directions(
        self, start_location: str, destination: str, context: RunContext
    ) -> str:
        """Gives walking directions between two campus locations using the map graph."""
        return await self.content_tools.get_campus_directions(
            start_location, destination, context
        )

    @function_tool
    async def ask_about_events(self, question: str, context: RunContext) -> str:
        """Answers questions about campus events using the vector database."""
        if not self.event_db:
            return "I'm sorry, the event database is not available right now."

        results = self.event_db.query_events(question)
        if not results:
            return "I couldn't find any specific events matching your question."

        context_str = "Found these relevant campus items:\n"
        for i, event in enumerate(results):
            category = event.get("category", "event")
            context_str += (
                f"{i + 1}. [{category}] {event.get('title', 'Item')} on "
                f"{event.get('date', 'Unknown Date')}: {event.get('description', '')}\n"
            )
        return context_str


def prewarm(proc: agents.JobProcess):
    """Heavy init runs once per worker process before any frontend connect."""
    print("Prewarming worker (image server, event DB, VAD)...")
    _start_voice_listener_once()
    _init_image_server()
    _build_event_db_sync()
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.3,
        prefix_padding_duration=0.2,
    )
    proc.userdata["image_server"] = _global_image_server
    proc.userdata["event_db"] = _global_event_db
    print("Prewarm complete — ready for instant LiveKit connect")


async def _monitor_presence_greetings(session: AgentSession, ctx: agents.JobContext):
    loop = asyncio.get_event_loop()

    def _blocking_get():
        return _presence_queue.get(timeout=1.0)

    while _session_live:
        try:
            await loop.run_in_executor(None, _blocking_get)
        except queue.Empty:
            continue

        if not _session_live:
            break
        if ctx.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            continue

        greeting = generate_presence_greeting()
        print(f"Presence greeting: {greeting}")
        _udp({"command": "wake"})
        _udp({"command": "conv_state", "state": "speaking", "emotion": "excited"})
        try:
            await session.say(greeting)
        except Exception as e:
            print(f"Presence greeting failed: {e}")


async def entrypoint(ctx: agents.JobContext):
    global _thinking_task, _awkward_timer_task, _smart_wait_task, _session_live

    print(f"Job received: room={ctx.room.name}")

    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        print("Warning: VAD not prewarmed, loading on connect (slow)")
        vad = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            prefix_padding_duration=0.2,
        )

    image_server = ctx.proc.userdata.get("image_server") or _global_image_server
    event_db = ctx.proc.userdata.get("event_db") or _global_event_db

    session = AgentSession(
        turn_handling=agents.TurnHandlingOptions(interruption={"mode": "vad"}),
        stt=deepgram.STT(model="nova-3"),
        tts=AmplitudeTTS(model="aura-2-luna-en"),
        vad=vad,
        llm=openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="openrouter/auto",
        ),
        tts_text_transforms=[
            "filter_markdown",
            "filter_emoji",
            filter_leaked_tool_syntax,
        ],
    )

    agent = CampusAgent(image_server, event_db)
    agent._room = ctx.room

    @ctx.room.on("data_received")
    def on_data_received(packet):
        try:
            payload = packet.data.decode("utf-8")
            data = json.loads(payload)
            if data.get("type") != "event_focus":
                return
            event = data.get("event", {})
            title = event.get("message") or event.get("title") or "this item"
            description = event.get("description", "")
            date = event.get("extracted_date") or event.get("date", "")
            location = event.get("extracted_location") or event.get("location", "")
            category = event.get("category", "event")
            detail_parts = []
            if date:
                detail_parts.append(f"on {date}")
            if location:
                detail_parts.append(f"at {location}")
            detail_str = " ".join(detail_parts)
            desc_str = f" {description}" if description else ""
            intro = (
                f"A visitor just tapped on the '{title}' {category} news card. "
                f"Tell them about this {category} enthusiastically."
            )
            if detail_str:
                intro += f" It is {detail_str}."
            intro += f"{desc_str} Then invite them to ask follow-up questions."
            print(f"Event focus received: {title} ({category})")
            asyncio.create_task(session.generate_reply(user_input=intro))
        except Exception as exc:
            print(f"event_focus handler error: {exc}")

    async def _hearing_reflex():
        _udp({"command": "wake"})
        _udp({"command": "conv_state", "state": "listening", "emotion": "excited"})
        await asyncio.sleep(0.4)
        _udp({"command": "conv_state", "state": "listening", "emotion": "attentive"})

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        if ev.new_state == "speaking":
            asyncio.create_task(_hearing_reflex())
        elif ev.new_state == "listening":
            global _smart_wait_task

            async def _smart_wait():
                await asyncio.sleep(1.2)
                if ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    _set_conv_state("waiting", "attentive")

            if _smart_wait_task and not _smart_wait_task.done():
                _smart_wait_task.cancel()
            _smart_wait_task = asyncio.create_task(_smart_wait())

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        global _thinking_task
        if not ev.is_final:
            return

        text = ev.transcript or ""
        junk = ["uh", "um", "ah", "er", "hmm", "okay", "so", "well"]
        clean_words = [w for w in text.lower().split() if w not in junk]
        word_count = len(clean_words)

        _thinking_task = asyncio.create_task(_thinking_cycle(word_count))

        try:
            _send_vader_emotion(text, is_agent=False)
        except Exception:
            pass

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        if ev.new_state == "speaking":
            _set_conv_state("speaking", "engaged")
        elif ev.new_state in ("listening", "idle"):
            _drain_to_zero()
            _set_conv_state("waiting", "attentive")

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        from livekit.agents.llm import ChatMessage

        if not isinstance(ev.item, ChatMessage):
            return
        text = ev.item.text_content or ""
        if ev.item.role == "assistant" and text:
            try:
                _send_vader_emotion(text, is_agent=True)
            except Exception as e:
                print("Vader Error:", e)

    print("Starting LiveKit session...")
    await session.start(room=ctx.room, agent=agent)
    asyncio.create_task(_monitor_presence_greetings(session, ctx))

    _session_live = True
    _udp({"command": "session_active", "active": True})
    try:
        await session.say("Oh hi! I am so happy you are talking to me!")
    except Exception as e:
        print(f"Initial greeting failed: {e}")

    try:
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
    finally:
        _session_live = False
        _udp({"command": "session_active", "active": False})


_listener_started = False


def _start_voice_listener_once():
    global _listener_started
    if not _listener_started:
        threading.Thread(target=_voice_udp_listener, daemon=True).start()
        _listener_started = True


if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="campus-greeting-agent",
            initialize_process_timeout=120,
            num_idle_processes=1,
        )
    )
