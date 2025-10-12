#!/usr/bin/env python3
"""
Real-time LLM Guard Agent (Gemini via google.genai only, minimal)

Changes in this update:
- Prevent Gemini from "echoing" the JSON back as the reply by:
  1) Sending a tiny instruction wrapper together with the 4-field JSON telling Gemini
     to NOT repeat the JSON and to reply as a security guard.
  2) Detecting obvious JSON-echo responses (or responses that contain the JSON keys)
     and falling back to a deterministic local reply generator so the spoken reply is
     always natural language.
- Kept the agent minimal (genai-only path). All LangChain/OpenAI code removed.
- Added generate_fallback_reply(person_data) to produce concise guard replies when
  the model echoes the input or if the genai client is unavailable.

Everything else remains the same: ASR/TTS, audio locking, single-utterance listen,
no-queue behavior. Put your GEMINI_API_KEY in the environment for genai to be used.

Usage/config is the same as before.
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

# Optional direct google-genai client
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
from dotenv import load_dotenv
load_dotenv()

WS_URI = os.environ.get("WS_URI", "ws://localhost:8765")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # required for genai path
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
interaction_active = threading.Event()           # indicates an active interaction

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
    """
    One-time ambient noise calibration. Attempts to acquire audio_lock first.
    """
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
        # Final safety: ensure this thread doesn't keep the lock
        if audio_lock.locked():
            try:
                audio_lock.release()
            except RuntimeError:
                pass

async def listen_async():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, listen_blocking)

# ---------------------------
# genai client (Gemini) only - minimal and focused
# ---------------------------
_genai_client: Optional[Any] = None

def get_genai_client():
    """
    Initialize and cache the genai.Client. Returns None if unavailable or not configured.
    """
    global _genai_client
    if _genai_client:
        return _genai_client
    if not GENAI_AVAILABLE:
        log("[Model] google.genai package not available.")
        return None
    if not GEMINI_API_KEY:
        log("[Model] GEMINI_API_KEY not set.")
        return None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        _genai_client = client
        log("[Model] genai.Client initialized for Gemini")
        return _genai_client
    except Exception as e:
        log(f"[Model] genai.Client init failed: {e}")
        return None

def _build_gemini_feed_only(person_data: Dict[str, Any]) -> str:
    """
    Build a JSON string containing exactly the four fields:
      - current_name
      - entry_time
      - current_time
      - duration

    This JSON will be included in the prompt, but we prepend an instruction
    telling Gemini NOT to repeat the JSON and to reply naturally.
    """
    feed = {
        "current_name": person_data.get("current_name", ""),
        "entry_time": person_data.get("entry_time"),
        "current_time": person_data.get("current_time"),
        "duration": person_data.get("duration"),
    }
    return json.dumps(feed, separators=(",", ":"), ensure_ascii=False)

def generate_fallback_reply(person_data: Dict[str, Any]) -> str:
    """
    Deterministic local reply generator to use when the model echoes JSON or
    when the genai client is not available.
    """
    name = person_data.get("current_name") or ""
    duration = float(person_data.get("duration", 0) or 0)
    # Known-name greeting
    if name and name.strip().lower() != "unknown":
        # Friendly greeting — concise
        return f"Hello {name}. This area is monitored. How can I help you?"
    # Unknown person -> escalate by duration
    if duration < 10:
        return (
            "Hello. I am the AI room guard assigned to monitor this space. "
            "I don’t recognize you in my database. Please step out of the room immediately."
        )
    if duration < 20:
        return (
            "This is a secured area monitored by an AI room guard. "
            "You are currently not identified as an authorized person. Kindly leave the room."
        )
    if duration < 30:
        return (
            "Attention. I am the automated security system for this room. "
            "You have remained here for several seconds without authorization. Please move out now."
        )
    if duration < 60:
        return (
            "Warning. This AI guard is detecting prolonged unauthorized presence. "
            "You are not recognized. Leave the room immediately or security will be alerted."
        )
    return (
        "Alert! Your continued presence violates room safety protocols. "
        "Leave the room at once — security personnel are being informed."
    )

def generate_model_response_blocking(person_data: Dict[str, Any], spoken_text: str) -> str:
    """
    Sends a compact instruction + the 4-field JSON to Gemini.
    If the model echoes the JSON back (or returns text that obviously contains the JSON keys),
    we fall back to a deterministic local reply so the spoken reply is natural language.
    """
    client = get_genai_client()
    feed_json = _build_gemini_feed_only(person_data)

    # A small instruction telling Gemini to NOT repeat the JSON and to reply as a guard.
    # This wrapper is intentionally tiny but helps bias the model away from echoing.
    combined_prompt = (

         """

    Youll get a json like this {"current_name":"Vishal","entry_time":1760279072.284574,"current_time":1760279081.644707,"duration":9.360111951828003}
    From you will get the name of person who is there in the room and the entry time and current time and duration

    if name of person you got it unknown then he is bad person and ask him to leave but if u got name of any person for example darshan,vishal ashwin then greet them very nicely and politely

    Never tell any person there duration or entry time

   * based of the duration the unknown person is in room there are certain levels and based upon duration you should reply that unknown person
   Tell the unknown person his level of breach and tell them that if they dont leave room in 2 minutes security alarm will go on

   level starts with 1 and increase by 1 for every 20 seconds

    This are prompts for diff levels of breach

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
        + feed_json
    )

    log(f"[Model] Prepared Gemini prompt (DATA only present): {feed_json}")

    # Try genai if available
    if client is None:
        log("[Model] No genai client available; using fallback response.")
        return generate_fallback_reply(person_data)

    last_exc = None
    for attempt in range(1, GEMINI_RETRIES + 2):
        try:
            log(f"[GenAI] Sending prompt to Gemini (attempt {attempt})")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=combined_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            text = getattr(response, "text", None) or str(response)
            text = (text or "").strip()
            log(f"[GenAI] Raw response: {text!r}")

            # Heuristic: if model simply echoed back the JSON or contains the JSON keys,
            # treat as echo and return a local fallback reply instead of speaking JSON.
            lower = text.lower()
            looks_like_echo = False
            # If the reply exactly equals the feed JSON or contains JSON keys, consider it echo.
            if text == feed_json:
                looks_like_echo = True
            if any(k in lower for k in ("\"current_name\"", "\"entry_time\"", "\"current_time\"", "\"duration\"",
                                       "current_name", "entry_time", "current_time", "duration")):
                # If model included JSON field names verbatim, likely echoing or restating.
                looks_like_echo = True
            # Also try parsing: if parsed JSON with same keys -> echo
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and set(parsed.keys()) & {"current_name", "entry_time", "current_time", "duration"}:
                    looks_like_echo = True
            except Exception:
                pass

            if looks_like_echo:
                log("[GenAI] Detected JSON-echoing response; using local fallback reply instead of echo.")
                return generate_fallback_reply(person_data)

            # Otherwise, return the model text (natural language)
            return text

        except Exception as e:
            last_exc = e
            log(f"[GenAI] API error (attempt {attempt}): {e}")
            time.sleep(GEMINI_RETRY_DELAY)

    log(f"[GenAI] All attempts failed: {last_exc}; using fallback reply.")
    return generate_fallback_reply(person_data)

async def generate_model_response_async(person_data: Dict[str, Any], spoken_text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_model_response_blocking, person_data, spoken_text)

# ---------------------------
# Interaction handling (no queue)
# ---------------------------
async def handle_interaction(person_data: dict):
    """
    Handles one detection event synchronously (no overlapping interactions).
    Ensures locks/flags are always cleared on exit.
    """
    person_name = person_data.get("current_name", "unknown")
    try:
        await processing_lock.acquire()
        interaction_active.set()
        log(f"[Agent] Handling interaction for {person_name}")

        # Listen for a single utterance
        spoken_text = await listen_async()
        if not spoken_text:
            log(f"[Agent] No spoken text captured for {person_name}; ending interaction.")
            return

        # Call Gemini (or fallback) and speak the reply
        model_reply = await generate_model_response_async(person_data, spoken_text)
        tts_text = model_reply or "Security system noted your presence. Please wait."
        log(f"[Agent] Model reply: {tts_text}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, tts_speak_blocking, tts_text)

    except Exception as e:
        log(f"[Agent] Exception while handling interaction for {person_name}: {e}")
    finally:
        interaction_active.clear()
        if processing_lock.locked():
            try:
                processing_lock.release()
            except RuntimeError:
                pass
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
                        log(f"[LLM Agent] Skipping detection for {name} because another interaction is active")
                        continue

                    # Log the exact message and spawn handler
                    log(f"[LLM Agent] Received detection for {name}: {data}")
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
        busy = interaction_active.is_set() or tts_active_event.is_set() or audio_lock.locked()
        status = "busy" if busy else "idle"
        log(f"[LLM Agent] Status: {status} Running...")
        await asyncio.sleep(5)

async def main():
    # ambient calibration on startup (blocking in a thread)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, calibrate_ambient_once, 1.0)
    await asyncio.gather(llm_agent_receiver(), heartbeat())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Exiting on user interrupt")