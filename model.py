#!/usr/bin/env python3
"""
Real-time LLM Guard Agent (no queue)

Behavior:
- Connects to a websocket feed (default ws://localhost:8765).
- On each detection message, if there's a person detected, the agent will
  (a) listen for a single user utterance (ASR),
  (b) call a LangChain-compatible model (OpenAI / Google Gemini if available) to
      respond as a SecurityGuard using a concise system prompt,
  (c) speak the model reply (TTS).
- There is no queue. If a new detection arrives while an interaction is in progress,
  the new detection is skipped (logged). This keeps the flow simple and real-time.
- The agent ensures no overlap between speaking and listening using an audio_lock
  and a tts_active_event plus conservative timing buffer around TTS playback.

Configuration (via environment):
- WS_URI (default ws://localhost:8765)
- OPENAI_API_KEY or GEMINI_API_KEY (if using LangChain ChatOpenAI / ChatGoogleGemini)
- TTS_RATE, TTS_VOLUME

Drop-in replacement for your previous agent when you want a simpler realtime behavior.
"""

import asyncio
import concurrent.futures
import json
import os
import time
import threading
from typing import Optional, Dict, Any

import pyttsx3
import speech_recognition as sr
import websockets

# Try LangChain chat model imports (best-effort); graceful fallback below.
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None
    LANGCHAIN_AVAILABLE = False

# Try ChatGoogleGemini wrapper (some LangChain installations expose this)
try:
    from langchain.chat_models import ChatGoogleGemini  # type: ignore
    LANGCHAIN_GG = True
except Exception:
    ChatGoogleGemini = None  # type: ignore
    LANGCHAIN_GG = False

# Optional direct google-genai fallback (if user had that previously)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GENAI_AVAILABLE = False

# ---------------------------
# CONFIG
# ---------------------------
import os
from dotenv import load_dotenv
load_dotenv()

WS_URI = os.environ.get("WS_URI", "ws://localhost:8765")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TTS_RATE = int(os.environ.get("TTS_RATE", 150))
TTS_VOLUME = float(os.environ.get("TTS_VOLUME", 1.0))

ASR_START_WAIT = 15
ASR_PHRASE_TIME_LIMIT = 30

RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0

MIC_DEVICE_INDEX = None

GEMINI_RETRIES = 2
GEMINI_RETRY_DELAY = 1.0

SPEECH_SAFETY_FACTOR = 1.10
MIN_SPEECH_PAD = 0.25

LOG_TIME_FMT = "%H:%M:%S"

# ---------------------------
# LOGGING
# ---------------------------
def log(msg: str):
    print(f"[{time.strftime(LOG_TIME_FMT)}] {msg}", flush=True)

# ---------------------------
# THREAD / ASYNC PRIMITIVES
# ---------------------------
audio_lock = threading.Lock()                    # protects microphone + speaker
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
tts_active_event = threading.Event()             # set while TTS is active
processing_lock = asyncio.Lock()                 # prevents overlapping interactions

# ---------------------------
# TTS
# ---------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)
tts_engine.setProperty("volume", TTS_VOLUME)

def _estimate_speech_duration_seconds(text: str, rate_words_per_min: float) -> float:
    words = len(text.split())
    if not rate_words_per_min or rate_words_per_min <= 0:
        rate_words_per_min = 150.0
    return (words / rate_words_per_min) * 60.0

def tts_speak_blocking(text: str, timeout_acquire: float = 5.0):
    """
    Blocking TTS. Sets tts_active_event and holds audio_lock while speaking.
    Sleeps a safety buffer after runAndWait to avoid ASR capturing TTS.
    """
    tts_active_event.set()
    got = audio_lock.acquire(timeout=timeout_acquire)
    if not got:
        log("[TTS] Could not acquire audio lock; skipping speak and clearing tts flag")
        tts_active_event.clear()
        return

    try:
        try:
            current_rate = tts_engine.getProperty("rate") or TTS_RATE
        except Exception:
            current_rate = TTS_RATE

        est_seconds = _estimate_speech_duration_seconds(text, float(current_rate))
        total_wait = max(0.0, est_seconds * SPEECH_SAFETY_FACTOR) + MIN_SPEECH_PAD

        log(f"[TTS] Speaking (est {est_seconds:.2f}s, hold {total_wait:.2f}s): {text}")
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            log(f"[TTS] Exception during runAndWait: {e}")

        # Safety sleep to avoid ASR capturing TTS audio
        try:
            time.sleep(total_wait)
        except Exception:
            pass

        log("[TTS] Finished speaking (and waited safety buffer)")
    finally:
        try:
            audio_lock.release()
        except RuntimeError:
            pass
        tts_active_event.clear()

# ---------------------------
# ASR
# ---------------------------
recognizer = sr.Recognizer()
ambient_calibrated = False
ambient_calibrate_lock = threading.Lock()

def calibrate_ambient_once(duration: float = 1.0):
    global ambient_calibrated
    with ambient_calibrate_lock:
        if ambient_calibrated:
            return
        got = audio_lock.acquire(timeout=5)
        if not got:
            log("[ASR] Could not acquire audio lock for ambient calibration")
            return
        try:
            with sr.Microphone(device_index=MIC_DEVICE_INDEX) as source:
                log("[ASR] Calibrating ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=duration)
                ambient_calibrated = True
                log("[ASR] Ambient calibration done")
        except Exception as e:
            log(f"[ASR] Ambient calibration failed: {e}")
        finally:
            try:
                audio_lock.release()
            except RuntimeError:
                pass

def listen_blocking(start_wait: float = ASR_START_WAIT, phrase_time_limit: int = ASR_PHRASE_TIME_LIMIT) -> Optional[str]:
    """
    Blocking listen. Waits for tts_active_event to clear, acquires audio_lock, listens,
    then releases audio_lock before performing network recognition.
    """
    # Wait for any current TTS activity to finish (but don't wait forever)
    waited = 0.0
    poll_interval = 0.05
    while tts_active_event.is_set() and waited < start_wait:
        time.sleep(poll_interval)
        waited += poll_interval
    if tts_active_event.is_set():
        log("[ASR] TTS active for too long; giving up on listening attempt")
        return None

    got_lock = audio_lock.acquire(timeout=start_wait)
    if not got_lock:
        log("[ASR] Could not acquire audio lock for listening (speaker busy or timeout)")
        return None

    try:
        with sr.Microphone(device_index=MIC_DEVICE_INDEX) as source:
            if not ambient_calibrated:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                except Exception as e:
                    log(f"[ASR] Ambient adjust failed in listen: {e}")

            log("[ASR] Listening for user...")
            audio = None
            try:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=phrase_time_limit)
                log("[ASR] Audio captured, sending to recognizer...")
            except sr.WaitTimeoutError:
                log("[ASR] No speech detected within timeout")
                return None
            except Exception as e:
                log(f"[ASR] Listening error: {e}")
                return None

        # Let other threads acquire audio_lock (TTS) while recognition happens
        try:
            audio_lock.release()
        except RuntimeError:
            pass

        try:
            text = recognizer.recognize_google(audio)
            log(f"[ASR] Heard: {text}")
            return text
        except sr.UnknownValueError:
            log("[ASR] Could not understand audio")
            return None
        except sr.RequestError as e:
            log(f"[ASR] ASR API error: {e}")
            return None
    finally:
        if audio_lock.locked():
            try:
                audio_lock.release()
            except RuntimeError:
                pass

async def listen_async():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, listen_blocking)

# ---------------------------
# Model prompt + LangChain integration
# ---------------------------
SYSTEM_PROMPT = """
if unknown_detected:
    duration = current_time - first_seen_time

    if name of person you got it unknown then he is abd person and ask him to leave and if u got name of any person for example darshan,vishal ahswin then greet them very nicely

    if duration < 10:
        agent.level = 1
    elif duration < 20:
        agent.level = 2
    elif duration < 30:
        agent.level = 3
    elif duration < 40:
        agent.level = 4
    else:
        agent.level = 5

    response = agent.respond("Unknown person still in view.")

    level_prompts = {
    1: (
        "Hello. I am the AI room guard assigned to monitor this space. "
        "I don’t recognize you in my database. Please step out of the room immediately."
    ),

    2: (
        "This is a secured area monitored by an AI room guard. "
        "You are currently not identified as an authorized person. "
        "Kindly leave the room for security reasons."
    ),

    3: (
        "Attention. I am the automated security system for this room. "
        "You have remained here for several seconds without authorization. "
        "Please move out of the room now before further action is taken."
    ),

    4: (
        "Warning. This AI guard is detecting prolonged unauthorized presence. "
        "You are not recognized in the trusted user list. "
        "Leave the room immediately or security will be alerted."
    ),

    5: (
        "Alert! I am the AI room security system, and your continued presence violates room safety protocols. "
        "You are not recognized. Leave the room at once — security personnel are being informed."
    ),
}
"""

def build_model_prompt(person_data: Dict[str, Any], spoken_text: str) -> str:
    feed_summary = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "current_name": person_data.get("current_name", ""),
        "duration_seconds": person_data.get("duration_seconds", 0),

    }
    prompt = (
        SYSTEM_PROMPT.strip()
        + "\n\nFEED:\n"
        + json.dumps(feed_summary, separators=(",", ":"), ensure_ascii=False)
        + "\n\nRECENT_SPEECH:\n"
        + json.dumps({"text": spoken_text or ""}, separators=(",", ":"), ensure_ascii=False)
        + "\n\nReply as the security guard (concise, natural language):"
    )
    return prompt

# Create LangChain chat model instance if available
_langchain_chat = None
def get_langchain_chat():
    global _langchain_chat
    if _langchain_chat:
        return _langchain_chat

    # Prefer LangChain ChatOpenAI (if available and OPENAI_API_KEY present)
    if LANGCHAIN_AVAILABLE and ChatOpenAI is not None and OPENAI_API_KEY:
        try:
            _langchain_chat = ChatOpenAI(temperature=0.2)
            log("[Model] Using ChatOpenAI via LangChain")
            return _langchain_chat
        except Exception as e:
            log(f"[Model] ChatOpenAI init failed: {e}")

    # Try LangChain ChatGoogleGemini if available
    if LANGCHAIN_GG and ChatGoogleGemini is not None and GEMINI_API_KEY:
        try:
            _langchain_chat = ChatGoogleGemini()
            log("[Model] Using ChatGoogleGemini via LangChain")
            return _langchain_chat
        except Exception as e:
            log(f"[Model] ChatGoogleGemini init failed: {e}")

    # Fallback to direct genai client if available
    if GENAI_AVAILABLE and GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            log("[Model] Using direct genai client as fallback")
            return client
        except Exception as e:
            log(f"[Model] genai client init failed: {e}")

    log("[Model] No suitable LangChain chat model available. Will use canned fallbacks.")
    return None

from typing import Dict, Any, List

# We'll keep a simple in-memory chat history per person
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}

def generate_model_response_blocking(person_data: Dict[str, Any], spoken_text: str, max_history: int = 10) -> str:
    """
    Blocking model call with chat history support.

    Important changes:
    - Uses build_model_prompt(...) (prompt_text) and sends it to all client paths so the
      model receives SYSTEM_PROMPT + FEED + RECENT_SPEECH.
    - For LangChain uses SystemMessage(prompt_text) as system instruction and maps assistant
      history to AIMessage.
    - For direct genai client sends contents=prompt_text.
    - Logs the prompt being sent for easier debugging.
    """
    person_id = person_data.get("id", "default")
    if person_id not in CHAT_HISTORY:
        CHAT_HISTORY[person_id] = []

    # Append user message to history (keeps local record)
    CHAT_HISTORY[person_id].append({"role": "user", "content": spoken_text or ""})

    # Limit history length
    history_to_use = CHAT_HISTORY[person_id][-max_history:]

    chat = get_langchain_chat()
    prompt_text = build_model_prompt(person_data, spoken_text)

    # Debug log the prompt we will send
    log(f"[Model] Prepared prompt for person_id={person_id} (len={len(prompt_text)} chars)")

    try:
        if chat is None:
            raise RuntimeError("no_model")

        # Direct genai client handling: send the full prompt_text
        if GENAI_AVAILABLE and isinstance(chat, genai.Client):
            last_exc = None
            for attempt in range(1, GEMINI_RETRIES + 2):
                try:
                    log(f"[GenAI] Querying genai (attempt {attempt}) with prompt length {len(prompt_text)}")
                    response = chat.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt_text,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        ),
                    )
                    text = getattr(response, "text", None) or str(response)
                    log("[GenAI] Response received")

                    # Append model response to history
                    CHAT_HISTORY[person_id].append({"role": "assistant", "content": text.strip()})
                    return text.strip()
                except Exception as e:
                    last_exc = e
                    log(f"[GenAI] API error: {e}")
                    time.sleep(GEMINI_RETRY_DELAY)
            log(f"[GenAI] All attempts failed: {last_exc}")
            raise RuntimeError("genai_failed")

        # LangChain Chat models handling: send the full prompt_text as system-level instruction
        if LANGCHAIN_AVAILABLE and SystemMessage is not None and HumanMessage is not None:
            try:
                # Use SystemMessage with the entire prompt_text (so FEED + RECENT_SPEECH are included)
                messages = [SystemMessage(content=prompt_text)]

                # Optionally add prior assistant/user exchanges as HumanMessage/AIMessage to preserve context.
                # This mapping requires AIMessage to be available in your import.
                for msg in history_to_use:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        # Use AIMessage (assistant) if available; otherwise fall back to SystemMessage
                        try:
                            from langchain.schema import AIMessage
                            messages.append(AIMessage(content=msg["content"]))
                        except Exception:
                            # Fallback: append as system-level content (less ideal)
                            messages.append(SystemMessage(content=msg["content"]))

                log("[LangChain] Sending messages to chat model (message count: %d)" % len(messages))
                response = chat(messages)
                # langchain ChatOpenAI returns an object with .content or may return a list
                text = getattr(response, "content", None)
                if text is None:
                    # Try common alternate shapes
                    try:
                        text = response[0].content
                    except Exception:
                        text = str(response)

                CHAT_HISTORY[person_id].append({"role": "assistant", "content": (text or "").strip()})
                return (text or "").strip()
            except Exception as e:
                log(f"[LangChain Chat] Error calling chat model: {e}")
                # fall through to canned fallback below

    except Exception as e:
        log(f"[Model] Model call failed or not available: {e}")
        return "I'm sorry, I cannot respond right now."

    # Canned fallback heuristic (unchanged)
    duration = float(person_data.get("duration_seconds", 0) or 0)
    behavior = person_data.get("behavior", {}) or {}
    aggression = float(person_data.get("aggression_score", 0) or 0.0)
    if aggression >= 0.6 or behavior.get("weapon_visible") or duration >= 60:
        return "This is an emergency. Leave the area immediately and wait for authorities to arrive."
    if duration >= 10 or behavior.get("loitering") or behavior.get("nervous_movement"):
        return "Hello — you are in a monitored area. Please identify yourself and state your purpose now."
    return "Hello. This area is monitored. Can I help you with something?"


    # Canned fallback heuristic
    duration = float(person_data.get("duration_seconds", 0) or 0)
    behavior = person_data.get("behavior", {}) or {}
    aggression = float(person_data.get("aggression_score", 0) or 0.0)
    if aggression >= 0.6 or behavior.get("weapon_visible") or duration >= 60:
        return "This is an emergency. Leave the area immediately and wait for authorities to arrive."
    if duration >= 10 or behavior.get("loitering") or behavior.get("nervous_movement"):
        return "Hello — you are in a monitored area. Please identify yourself and state your purpose now."
    return "Hello. This area is monitored. Can I help you with something?"

async def generate_model_response_async(person_data: Dict[str, Any], spoken_text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_model_response_blocking, person_data, spoken_text)

# ---------------------------
# Interaction handling (no queue)
# ---------------------------
async def handle_interaction(person_data: dict):
    """
    Handles one detection event synchronously (no overlapping interactions).
    """
    person_name = person_data.get("current_name", "unknown")
    try:
        # Acquire processing lock to prevent overlap
        await processing_lock.acquire()
        log(f"[Agent] Handling interaction for {person_name}")

        # Listen for an utterance
        spoken_text = await listen_async()
        if not spoken_text:
            log(f"[Agent] No spoken text captured for {person_name}; releasing lock.")
            return

        # Get model response and speak it
        model_reply = await generate_model_response_async(person_data, spoken_text)
        tts_text = model_reply or "Security system noted your presence. Please wait."
        log(f"[Agent] Model reply: {tts_text}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, tts_speak_blocking, tts_text)

    except Exception as e:
        log(f"[Agent] Exception while handling interaction for {person_name}: {e}")
    finally:
        # Ensure lock is released
        if processing_lock.locked():
            processing_lock.release()
        log(f"[Agent] Finished interaction for {person_name}")

# ---------------------------
# Websocket receiver (simple, no queue)
# ---------------------------
async def llm_agent_receiver():
    backoff = RECONNECT_BASE_DELAY
    while True:
        try:
            log(f"[LLM Agent] Connecting to {WS_URI} ...")
            async with websockets.connect(WS_URI) as ws:
                log(f"[LLM Agent] Connected to {WS_URI}")
                backoff = RECONNECT_BASE_DELAY
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        log("[LLM Agent] Received non-JSON message; ignoring")
                        continue

                    name = data.get("current_name")
                    if not name:
                        # no person detected; ignore
                        continue

                    # If already handling an interaction, skip new detections to keep things simple
                    if processing_lock.locked():
                        continue

                    # Fire off a task to handle interaction (it will acquire the lock)
                    asyncio.create_task(handle_interaction(data))

        except (websockets.ConnectionClosedError, ConnectionRefusedError) as e:
            log(f"[LLM Agent] Websocket connection error: {e}. Reconnecting in {backoff}s...")
        except Exception as e:
            log(f"[LLM Agent] Unexpected error in websocket receiver: {e}. Reconnecting in {backoff}s...")

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, RECONNECT_MAX_DELAY)

# ---------------------------
# Main / startup
# ---------------------------
async def heartbeat():
    while True:
        # Show whether we're busy
        status = "busy" if processing_lock.locked() else "idle"
        log(f"[LLM Agent] Status: {status} Running...")
        await asyncio.sleep(5)

async def main():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, calibrate_ambient_once, 1.0)
    await asyncio.gather(llm_agent_receiver(), heartbeat())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Exiting on user interrupt")