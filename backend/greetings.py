"""Time-aware greeting text for proactive hellos."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict

_last_seen: Dict[str, datetime] = {}


def get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def get_context(name: str) -> str:
    now = datetime.now()
    if name not in _last_seen:
        return "first_time"
    diff = (now - _last_seen[name]).total_seconds()
    if diff < 300:
        return "same_session"
    if diff < 3600:
        return "short_break"
    return "returning"


def mark_seen(name: str) -> None:
    _last_seen[name] = datetime.now()


def generate_greeting(name: str, is_known: bool = True) -> str:
    time_of_day = get_time_of_day()

    if not is_known or name == "Unknown":
        unknown_greetings = [
            "Hi there! I don't think we've met. What's your name?",
            "Hey! I don't recognize you yet. Mind introducing yourself?",
            "Hello! I'm not sure we've been introduced. What should I call you?",
        ]
        return random.choice(unknown_greetings)

    context = get_context(name)
    mark_seen(name)

    if context == "same_session":
        return random.choice(
            [
                f"Oh, {name}! Back again?",
                f"Hey {name}, you're back!",
                f"Welcome back, {name}!",
            ]
        )

    if context == "short_break":
        return random.choice(
            [
                f"Hey {name}! How's it going?",
                f"Oh, {name}! Good to see you again.",
                f"{name}! What's up?",
            ]
        )

    if context == "returning":
        if time_of_day == "morning":
            pool = [f"Good morning, {name}!", f"Morning, {name}! How are you?"]
        elif time_of_day == "afternoon":
            pool = [f"Good afternoon, {name}!", f"Hey {name}! How's your day going?"]
        elif time_of_day == "evening":
            pool = [f"Good evening, {name}!", f"Hey {name}! Still around this evening?"]
        else:
            pool = [f"Hey {name}! Working late?", f"{name}! You're here late."]
        return random.choice(pool)

    if time_of_day == "morning":
        pool = [f"Good morning, {name}!", f"Hey {name}! Good morning!"]
    elif time_of_day == "afternoon":
        pool = [f"Good afternoon, {name}!", f"Hey {name}! Good to see you!"]
    elif time_of_day == "evening":
        pool = [f"Good evening, {name}!", f"Hey {name}! Nice to see you!"]
    else:
        pool = [f"Hey {name}!", f"{name}! What brings you here?"]
    return random.choice(pool)


def generate_presence_greeting() -> str:
    """Generic greeting when someone approaches during an active voice session (no face ID)."""
    time_of_day = get_time_of_day()
    if time_of_day == "morning":
        pool = [
            "Oh! Good morning! I didn't expect you — hi!",
            "Hey there! Good morning! Come say hi!",
            "Morning! I see you over there!",
        ]
    elif time_of_day == "afternoon":
        pool = [
            "Hey! Good afternoon! I'm so happy you're here!",
            "Oh hi! I see you — come talk to me!",
            "Hello there! Good afternoon!",
        ]
    elif time_of_day == "evening":
        pool = [
            "Hey! Good evening! Want to chat?",
            "Oh, hi! I see you — good evening!",
            "Hello! Evening's a great time to talk!",
        ]
    else:
        pool = [
            "Hey! You're here late — hi!",
            "Oh hi! I didn't expect anyone this late!",
            "Hello there! Working late too?",
        ]
    return random.choice(pool)
