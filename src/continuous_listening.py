# continuous_listening.py - Always-on voice pipeline for JARVIS
"""
VAD -> Whisper -> Intent classification pipeline.
Runs in background thread, queues events for main thread.
"""

import threading
import queue
import time
import re
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

# Audio
import sounddevice as sd


class Intent(Enum):
    IDENTIFY = "identify"      # "what's this?", "what is that?"
    CORRECT = "correct"        # "no, that's a...", "actually it's..."
    CONFIRM = "confirm"        # "yes", "right", "correct"
    TEACH = "teach"            # "this is called...", "this is a..."
    IGNORE = "ignore"          # no relevant intent detected


@dataclass
class VoiceEvent:
    """Event produced when speech with relevant intent is detected"""
    intent: Intent
    transcript: str
    timestamp: float          # when speech ended
    audio_duration: float     # how long the utterance was
    extracted_name: Optional[str] = None  # parsed object name (for TEACH/CORRECT)


# =============================================================================
# TRANSCRIPT PARSING
# =============================================================================

def extract_object_name(transcript: str) -> Optional[str]:
    """
    Extract object name from TEACH intent.

    Examples:
        "this is called a multimeter" -> "multimeter"
        "this is a Phillips head" -> "Phillips head"
        "that's a soldering iron" -> "soldering iron"
    """
    transcript = transcript.lower().strip()

    # Ordered by specificity (longer prefixes first)
    prefixes = [
        "this is called a ",
        "this is called an ",
        "this is called ",
        "that is called a ",
        "that is called an ",
        "that is called ",
        "it's called a ",
        "it's called an ",
        "it's called ",
        "this is a ",
        "this is an ",
        "this is my ",
        "this is ",
        "that's a ",
        "that's an ",
        "that's my ",
        "that's ",
        "that is a ",
        "that is an ",
        "that is ",
        "it's a ",
        "it's an ",
        "it's my ",
        "it's ",
    ]

    # Non-object words/phrases that should not be extracted as object names
    non_objects = {
        "impossible", "correct", "right", "wrong", "good", "bad", "great",
        "nice", "fine", "okay", "ok", "true", "false", "working", "broken",
        "amazing", "terrible", "perfect", "weird", "strange", "interesting",
        "cool", "awesome", "fantastic", "horrible", "ridiculous", "stupid",
    }

    for prefix in prefixes:
        if transcript.startswith(prefix):
            name = transcript[len(prefix):].strip()
            # Clean trailing punctuation
            name = name.rstrip('.,!?')

            # Reject non-object words
            if name in non_objects:
                return None

            # Reject negations (e.g., "not a hammer")
            if name.startswith("not "):
                return None

            return name if len(name) >= 2 else None

    return None


def extract_correction(transcript: str) -> Optional[str]:
    """
    Extract corrected name from CORRECT intent.

    Examples:
        "no, that's a flathead" -> "flathead"
        "actually it's a resistor" -> "resistor"
        "no that's my Phillips screwdriver" -> "Phillips screwdriver"

    NOT valid (denials without new name):
        "no, that's not a hammer" -> None (denial, not providing new name)
        "that's not a pair of scissors" -> None
    """
    transcript = transcript.lower().strip()

    # Strip correction prefixes
    correction_prefixes = [
        "no, ",
        "no ",
        "nope, ",
        "nope ",
        "actually, ",
        "actually ",
        "wait, ",
        "wait ",
    ]

    for prefix in correction_prefixes:
        if transcript.startswith(prefix):
            transcript = transcript[len(prefix):]
            break

    # Check for denial patterns - these are NOT corrections with a new name
    # e.g., "that's not a hammer" = denying, not providing new name
    denial_patterns = [
        "that's not ",
        "thats not ",
        "that is not ",
        "it's not ",
        "its not ",
        "it is not ",
        "this is not ",
        "this isn't ",
        "that isn't ",
        "it isn't ",
    ]

    for pattern in denial_patterns:
        if transcript.startswith(pattern):
            return None  # Denial without new name

    # Now apply TEACH extraction on remainder
    return extract_object_name(transcript)


def normalize_transcript(text: str) -> str:
    """Normalize transcript for intent matching"""
    text = text.lower().strip()
    # Remove punctuation for matching
    text = re.sub(r'[.,!?;:]', '', text)
    return text


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

def classify_intent(transcript: str) -> Tuple[Intent, Optional[str]]:
    """
    Classify intent and extract object name if applicable.
    Returns (intent, extracted_name or None)
    """
    normalized = normalize_transcript(transcript)
    original_lower = transcript.lower().strip()

    # CONFIRM - check first (short responses)
    confirm_phrases = {
        "yes", "yeah", "yep", "right", "correct",
        "thats right", "that's right",
        "thats correct", "that's correct",
        "yup", "uh huh", "mm hmm", "affirmative"
    }
    if normalized in confirm_phrases:
        return Intent.CONFIRM, None

    # IDENTIFY - "what's this?", "what is that?"
    if normalized.startswith("whats") or normalized.startswith("what is") or normalized.startswith("what's"):
        # Must contain "this" or "that"
        if " this" in normalized or " that" in normalized or normalized.endswith("this") or normalized.endswith("that"):
            return Intent.IDENTIFY, None

    # CORRECT - "no, that's a...", "actually it's..."
    correction_starters = ["no ", "no,", "nope ", "nope,", "actually ", "actually,",
                          "that's not", "thats not", "it's not", "its not", "wait "]
    for starter in correction_starters:
        if normalized.startswith(starter):
            extracted = extract_correction(original_lower)
            if extracted:
                return Intent.CORRECT, extracted
            # Started with correction but couldn't parse name
            # Could be "no that's not right" (complaint, not correction)
            break

    # TEACH - "this is called...", "this is a..."
    teach_starters = ["this is", "that's a", "thats a", "that is a", "it's called",
                      "its called", "this is called", "that's my", "thats my", "this is my"]
    for starter in teach_starters:
        if normalized.startswith(starter):
            extracted = extract_object_name(original_lower)
            if extracted:
                return Intent.TEACH, extracted
            break

    # IGNORE - everything else
    return Intent.IGNORE, None


# =============================================================================
# CONTINUOUS LISTENER
# =============================================================================

class ContinuousListener:
    """
    Always-on voice pipeline: VAD -> Whisper -> Intent classification

    Usage:
        listener = ContinuousListener(
            whisper_model=shared_whisper,
            event_queue=event_queue
        )
        listener.start()
        # ... main loop polls event_queue ...
        listener.stop()
    """

    def __init__(
        self,
        whisper_model,                # Shared Whisper model (not loaded here)
        event_queue: queue.Queue,     # Events go here for main thread
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,   # Silero confidence threshold
        silence_duration: float = 0.8 # Seconds of silence to end utterance
    ):
        self.whisper = whisper_model
        self.event_queue = event_queue
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._vad_model = None

        # Audio state
        self.chunk_samples = 512  # 32ms at 16kHz (minimum for Silero VAD)
        self.chunk_duration = self.chunk_samples / self.sample_rate

        # Concurrency control - only allow one Whisper call at a time
        self._whisper_lock = threading.Lock()
        self._processing = False  # Flag to skip overlapping transcriptions

    def start(self):
        """Start listening in background thread"""
        if self._running:
            return

        print("  Starting continuous listener...")
        self._running = True
        self._load_vad()
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        print("  Continuous listener started")

    def stop(self):
        """Stop listening, cleanup resources"""
        print("  Stopping continuous listener...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("  Continuous listener stopped")

    def _load_vad(self):
        """Load Silero VAD model"""
        print("    Loading Silero VAD...")
        self._vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self._vad_model.eval()
        print("    VAD loaded")

    def _audio_loop(self):
        """Main audio capture and processing loop"""
        # Audio buffer for current utterance
        audio_buffer = []
        is_speaking = False
        silence_chunks = 0
        silence_chunks_threshold = int(self.silence_duration / self.chunk_duration)

        # For VAD state reset
        self._vad_model.reset_states()

        def audio_callback(indata, frames, time_info, status):
            """Called by sounddevice for each audio chunk"""
            nonlocal audio_buffer, is_speaking, silence_chunks

            if status:
                print(f"    Audio status: {status}")

            # Convert to format VAD expects (float32, mono)
            audio_chunk = indata[:, 0].copy()  # Take first channel

            # Convert to torch tensor for VAD
            audio_tensor = torch.from_numpy(audio_chunk).float()

            # Run VAD
            try:
                speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()
            except Exception as e:
                print(f"    VAD error: {e}")
                return

            is_speech = speech_prob > self.vad_threshold

            if is_speech:
                if not is_speaking:
                    # Speech started
                    is_speaking = True
                    silence_chunks = 0
                    audio_buffer = []

                audio_buffer.append(audio_chunk)
                silence_chunks = 0

            else:
                if is_speaking:
                    # Still in utterance, but silence detected
                    audio_buffer.append(audio_chunk)  # Include some silence
                    silence_chunks += 1

                    if silence_chunks >= silence_chunks_threshold:
                        # Speech ended - process utterance
                        is_speaking = False

                        if len(audio_buffer) > 10:  # Minimum length check (~320ms)
                            # Skip if already processing (avoid piling up Whisper calls)
                            if self._processing:
                                print("    (skipping - still processing previous)")
                            else:
                                speech_end_time = time.time()
                                try:
                                    full_audio = np.concatenate(audio_buffer)

                                    # Process in separate thread to not block audio
                                    threading.Thread(
                                        target=self._process_utterance,
                                        args=(full_audio.copy(), speech_end_time),
                                        daemon=True
                                    ).start()
                                except Exception as e:
                                    print(f"    Audio processing error: {e}")

                        audio_buffer = []
                        silence_chunks = 0

                        # Reset VAD state for next utterance
                        try:
                            self._vad_model.reset_states()
                        except Exception as e:
                            print(f"    VAD reset error: {e}")

        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_samples,
                callback=audio_callback
            ):
                print("    Audio stream started")
                while self._running:
                    time.sleep(0.1)

        except Exception as e:
            print(f"    Audio stream error: {e}")
            self._running = False

    def _process_utterance(self, audio_buffer: np.ndarray, speech_end_time: float):
        """Process completed utterance - transcribe and classify"""
        # Set processing flag to prevent piling up
        self._processing = True
        try:
            # Ensure float32 in correct range
            if audio_buffer.dtype != np.float32:
                audio_buffer = audio_buffer.astype(np.float32)

            # Normalize if needed
            max_val = np.abs(audio_buffer).max()
            if max_val > 1.0:
                audio_buffer = audio_buffer / max_val

            # Skip very short audio (likely noise)
            if len(audio_buffer) < self.sample_rate * 0.3:  # Less than 300ms
                return

            # Transcribe
            try:
                result = self.whisper.transcribe(
                    audio_buffer,
                    language="en",
                    fp16=False,  # GTX 1060 compatibility
                    no_speech_threshold=0.6,  # Higher = more aggressive silence detection
                    condition_on_previous_text=False,  # Reduces hallucination chaining
                )
                transcript = result["text"].strip()
            except Exception as e:
                print(f"    Whisper error: {e}")
                return

            # Skip empty or very short transcripts
            if not transcript or len(transcript) < 2:
                return

            # Filter common Whisper hallucinations
            hallucination_phrases = [
                "thanks for watching",
                "see you next time",
                "subscribe",
                "like and subscribe",
                "bye bye",
                "have a great",
                "thank you for",
                "please subscribe",
                "don't forget to",
            ]
            transcript_lower = transcript.lower()
            if any(phrase in transcript_lower for phrase in hallucination_phrases):
                print(f"    (filtered hallucination: '{transcript}')")
                return

            # Classify intent
            intent, extracted_name = classify_intent(transcript)

            # Log for debugging
            print(f"    Heard: '{transcript}' -> {intent.value}", end="")
            if extracted_name:
                print(f" [{extracted_name}]")
            else:
                print()

            if intent == Intent.IGNORE:
                return

            # Queue event for main thread
            event = VoiceEvent(
                intent=intent,
                transcript=transcript,
                timestamp=speech_end_time,
                audio_duration=len(audio_buffer) / self.sample_rate,
                extracted_name=extracted_name
            )
            self.event_queue.put(event)

        except Exception as e:
            print(f"    Transcription error: {e}")
        finally:
            # Always clear processing flag
            self._processing = False


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    import whisper

    print("=" * 60)
    print("Continuous Listening Test")
    print("=" * 60)

    # Load Whisper
    print("\nLoading Whisper...")
    whisper_model = whisper.load_model("small")
    print("Whisper loaded")

    # Create event queue
    event_queue = queue.Queue()

    # Create and start listener
    listener = ContinuousListener(
        whisper_model=whisper_model,
        event_queue=event_queue,
        vad_threshold=0.5,
        silence_duration=0.8
    )

    listener.start()

    print("\n" + "=" * 60)
    print("Listening... Speak to test intents.")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            try:
                event = event_queue.get(timeout=1.0)
                print(f"\n>>> EVENT: {event.intent.value}")
                print(f"    Transcript: {event.transcript}")
                if event.extracted_name:
                    print(f"    Extracted name: {event.extracted_name}")
                print(f"    Duration: {event.audio_duration:.2f}s")
                print()
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print("\n\nStopping...")
        listener.stop()
        print("Done")
