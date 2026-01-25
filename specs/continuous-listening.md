# Continuous Listening Spec

*Status: Ready for implementation*
*Last updated: 2026-01-24*

## Goal

Natural voice interaction without button presses or wake words. Enables faster episode accumulation through conversational flow.

## Prerequisites

Before implementing this module, the following changes to `arnold_integrated_v2.py` are required:

1. **Rolling visual buffer** - Background thread continuously capturing frames with timestamps
2. **Shared Whisper model** - Extract from VoiceInterface for reuse
3. **Last recognition state** - Track most recent identification for corrections/confirmations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Background Capture Thread                        │
│  ┌─────────┐    ┌─────────────┐    ┌───────────────────────┐   │
│  │ Camera  │───▶│  V-JEPA     │───▶│  Rolling Buffer       │   │
│  │ Stream  │    │ (embedding) │    │  (frame, emb, time)   │   │
│  └─────────┘    └─────────────┘    └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Audio Thread (always running)                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│  │  Mic    │───▶│  VAD    │───▶│ Buffer  │                     │
│  │ Stream  │    │(Silero) │    │(on speech)                    │
│  └─────────┘    └─────────┘    └────┬────┘                     │
│                                     │ speech_end               │
│                                     ▼                          │
│                              ┌─────────────┐                   │
│                              │  Whisper    │                   │
│                              │ (shared)    │                   │
│                              └──────┬──────┘                   │
└─────────────────────────────────────┼──────────────────────────┘
                                      │ transcript
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Intent Router                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  "what's this" / "what is that"  → IDENTIFY            │    │
│  │  "no, that's a..." / "actually"  → CORRECT             │    │
│  │  "yes" / "right" / "correct"     → CONFIRM             │    │
│  │  "this is called..." / "this is a" → TEACH             │    │
│  │  (no match)                      → IGNORE              │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────┬──────────────────────────┘
                                      │ (intent, transcript)
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Temporal Binder                                     │
│  • Grabs visual context from rolling buffer                     │
│  • ~1-2 sec lookback (user was looking at object BEFORE speech) │
│  • Thread-safe access with Lock                                 │
│  • Packages (frame, embedding, transcript, intent)              │
└─────────────────────────────────────┬──────────────────────────┘
                                      │
                                      ▼
                            Main thread handlers
                         (queued via thread-safe queue)
```

## Components

### 1. Dependencies

```bash
pip install silero-vad sounddevice
# Whisper already available via existing VoiceInterface
```

Note: Standardizing on `sounddevice` over `pyaudio` for simpler API.

### 2. File: `continuous_listening.py`

New module, standalone audio pipeline.

#### Data structures

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

class Intent(Enum):
    IDENTIFY = "identify"      # "what's this?", "what is that?"
    CORRECT = "correct"        # "no, that's a...", "actually it's..."
    CONFIRM = "confirm"        # "yes", "right", "correct"
    TEACH = "teach"            # "this is called...", "this is a..."
    IGNORE = "ignore"          # no relevant intent detected

@dataclass
class VoiceEvent:
    intent: Intent
    transcript: str
    timestamp: float          # when speech ended
    audio_duration: float     # how long the utterance was
    extracted_name: Optional[str] = None  # parsed object name (for TEACH/CORRECT)

@dataclass
class VisualContext:
    """Visual context retrieved from rolling buffer"""
    frame: np.ndarray
    embedding: torch.Tensor
    timestamp: float
    detections: list  # List[Detection] from most recent Florence run
```

#### Class: `ContinuousListener`

```python
import threading
import queue

class ContinuousListener:
    """
    Always-on voice pipeline: VAD -> Whisper -> Intent classification

    Usage:
        listener = ContinuousListener(
            whisper_model=shared_whisper,  # Pass existing model
            event_queue=event_queue        # Thread-safe queue for events
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

    def start(self):
        """Start listening in background thread"""
        if self._running:
            return
        self._running = True
        self._load_vad()
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening, cleanup resources"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _load_vad(self):
        """Load Silero VAD model"""
        ...

    def _audio_loop(self):
        """Main audio capture and processing loop"""
        ...

    def _process_utterance(self, audio_buffer: np.ndarray, speech_end_time: float):
        """Transcribe and classify, then queue event"""
        ...

    def _classify_intent(self, transcript: str) -> Tuple[Intent, Optional[str]]:
        """
        Classify intent and extract object name if applicable.
        Returns (intent, extracted_name or None)
        """
        ...
```

#### Intent classification rules (v1)

Simple keyword matching - upgrade to classifier later if needed.

| Intent | Patterns | Extraction |
|--------|----------|------------|
| IDENTIFY | starts with "what's", "what is"; contains "this" or "that" | None |
| CORRECT | starts with "no", "actually", "that's not", "it's not" | Strip prefix, apply TEACH extraction |
| CONFIRM | entire transcript is "yes", "yeah", "yep", "right", "correct", "that's right" | None |
| TEACH | starts with "this is", "that's a", "that is a", "it's called", "this is called" | Strip prefix, remainder is object name |
| IGNORE | everything else | None |

Normalize transcript to lowercase, strip punctuation before matching.

#### Transcript parsing helpers

```python
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

    for prefix in prefixes:
        if transcript.startswith(prefix):
            name = transcript[len(prefix):].strip()
            # Clean trailing punctuation
            name = name.rstrip('.,!?')
            return name if len(name) >= 2 else None

    return None


def extract_correction(transcript: str) -> Optional[str]:
    """
    Extract corrected name from CORRECT intent.

    Examples:
        "no, that's a flathead" -> "flathead"
        "actually it's a resistor" -> "resistor"
        "no that's my Phillips screwdriver" -> "Phillips screwdriver"
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

    # Now apply TEACH extraction on remainder
    return extract_object_name(transcript)
```

### 3. VAD setup

Silero VAD via torch hub:

```python
def _load_vad(self):
    """Load Silero VAD for streaming"""
    self._vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    self._vad_model.eval()

    # For streaming, we'll process 30ms chunks
    # 30ms at 16kHz = 480 samples
    self.chunk_samples = 480
```

For streaming implementation:
- Process 30ms chunks (480 samples at 16kHz)
- Track speech state transitions
- Buffer audio while `is_speech=True`
- On transition to `is_speech=False` for `silence_duration`, emit buffered audio

### 4. Whisper setup (shared model)

The main `WorkshopArnold` class will own the Whisper model:

```python
# In WorkshopArnold.__init__
self.whisper_model = whisper.load_model("base")  # Shared instance
self.voice = VoiceInterface(whisper_model=self.whisper_model)  # Pass to existing
self.listener = ContinuousListener(
    whisper_model=self.whisper_model,  # Pass to listener
    event_queue=self.voice_event_queue
)
```

Transcription (uses shared model):

```python
def _process_utterance(self, audio_buffer: np.ndarray, speech_end_time: float):
    """Process completed utterance"""
    # Normalize audio to float32 [-1, 1]
    if audio_buffer.dtype == np.int16:
        audio_float = audio_buffer.astype(np.float32) / 32768.0
    else:
        audio_float = audio_buffer

    # Transcribe (brief VRAM spike, but V-JEPA not running simultaneously)
    result = self.whisper.transcribe(
        audio_float,
        language="en",
        fp16=False  # GTX 1060 compatibility
    )
    transcript = result["text"].strip()

    if not transcript:
        return

    # Classify intent and extract name
    intent, extracted_name = self._classify_intent(transcript)

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
```

### 5. Rolling Visual Buffer (new component)

Add to `arnold_integrated_v2.py`:

```python
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
import time

@dataclass
class BufferedFrame:
    """Single frame with metadata in rolling buffer"""
    frame: np.ndarray
    embedding: torch.Tensor
    timestamp: float
    detections: List[Detection]  # Florence detections for this frame


class RollingVisualBuffer:
    """
    Thread-safe rolling buffer of recent frames with embeddings.

    Runs continuous capture in background thread.
    Main thread and voice thread can safely query for recent frames.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        vjepa_encoder: VJEPAEncoder,
        florence_detector: FlorenceDetector,
        buffer_seconds: float = 3.0,
        target_fps: float = 10.0,  # Don't need 30fps for this
        detection_interval: int = 5  # Run Florence every N frames
    ):
        self.cap = cap
        self.vjepa = vjepa_encoder
        self.florence = florence_detector

        self.buffer_size = int(buffer_seconds * target_fps)
        self.frame_interval = 1.0 / target_fps
        self.detection_interval = detection_interval

        self._buffer: deque[BufferedFrame] = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Cache latest detections (don't run Florence every frame)
        self._latest_detections: List[Detection] = []
        self._frame_count = 0

    def start(self):
        """Start background capture"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background capture"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _capture_loop(self):
        """Background capture and embedding loop"""
        frame_buffer = []  # Accumulate frames for V-JEPA (needs multiple)

        while self._running:
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)

            # Keep last 16 frames for V-JEPA (minimal for embedding)
            if len(frame_buffer) > 16:
                frame_buffer.pop(0)

            # Run Florence periodically
            self._frame_count += 1
            if self._frame_count % self.detection_interval == 0:
                self._latest_detections = self.florence.detect(frame_rgb)

            # Generate embedding when we have enough frames
            if len(frame_buffer) >= 16:
                frames_array = np.array(frame_buffer)
                embedding = self.vjepa.encode_frames(frames_array)

                buffered = BufferedFrame(
                    frame=frame_rgb.copy(),
                    embedding=embedding.cpu(),
                    timestamp=time.time(),
                    detections=self._latest_detections.copy()
                )

                with self._lock:
                    self._buffer.append(buffered)

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame_at_time(self, target_time: float) -> Optional[BufferedFrame]:
        """
        Get the frame closest to target_time.
        Thread-safe.
        """
        with self._lock:
            if not self._buffer:
                return None

            # Find closest frame
            closest = min(self._buffer, key=lambda f: abs(f.timestamp - target_time))

            # Return a copy to avoid threading issues
            return BufferedFrame(
                frame=closest.frame.copy(),
                embedding=closest.embedding.clone(),
                timestamp=closest.timestamp,
                detections=closest.detections.copy()
            )

    def get_latest(self) -> Optional[BufferedFrame]:
        """Get most recent frame. Thread-safe."""
        with self._lock:
            if not self._buffer:
                return None
            latest = self._buffer[-1]
            return BufferedFrame(
                frame=latest.frame.copy(),
                embedding=latest.embedding.clone(),
                timestamp=latest.timestamp,
                detections=latest.detections.copy()
            )

    def get_frames_in_range(self, start_time: float, end_time: float) -> List[BufferedFrame]:
        """Get all frames in time range. Thread-safe."""
        with self._lock:
            return [
                BufferedFrame(
                    frame=f.frame.copy(),
                    embedding=f.embedding.clone(),
                    timestamp=f.timestamp,
                    detections=f.detections.copy()
                )
                for f in self._buffer
                if start_time <= f.timestamp <= end_time
            ]
```

### 6. Last Recognition State

Add to `WorkshopArnold`:

```python
@dataclass
class LastRecognition:
    """Tracks the most recent identification for corrections/confirmations"""
    name: str                    # What we identified it as
    category: str                # Florence category
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float
    embedding: torch.Tensor      # For correction training


class WorkshopArnold:
    def __init__(self, ...):
        ...
        # Last recognition state (for corrections/confirmations)
        self.last_recognition: Optional[LastRecognition] = None
        self._last_recognition_lock = threading.Lock()

    def _set_last_recognition(self, name: str, category: str, confidence: float,
                               bbox: Tuple, embedding: torch.Tensor):
        """Thread-safe update of last recognition"""
        with self._last_recognition_lock:
            self.last_recognition = LastRecognition(
                name=name,
                category=category,
                confidence=confidence,
                bbox=bbox,
                timestamp=time.time(),
                embedding=embedding.clone()
            )

    def _get_last_recognition(self) -> Optional[LastRecognition]:
        """Thread-safe read of last recognition"""
        with self._last_recognition_lock:
            if self.last_recognition is None:
                return None
            # Return copy
            return LastRecognition(
                name=self.last_recognition.name,
                category=self.last_recognition.category,
                confidence=self.last_recognition.confidence,
                bbox=self.last_recognition.bbox,
                timestamp=self.last_recognition.timestamp,
                embedding=self.last_recognition.embedding.clone()
            )
```

### 7. Integration with main system

Updated integration in `arnold_integrated_v2.py`:

```python
import queue
from continuous_listening import ContinuousListener, Intent, VoiceEvent

class WorkshopArnold:
    def __init__(self, ...):
        ...
        # Voice event queue (thread-safe communication)
        self.voice_event_queue: queue.Queue[VoiceEvent] = queue.Queue()

        # Shared Whisper model
        self.whisper_model = whisper.load_model("base")

        # Rolling visual buffer (started after camera init)
        self.visual_buffer: Optional[RollingVisualBuffer] = None

        # Continuous listener
        self.listener = ContinuousListener(
            whisper_model=self.whisper_model,
            event_queue=self.voice_event_queue
        )

        # Last recognition state
        self.last_recognition: Optional[LastRecognition] = None
        self._last_recognition_lock = threading.Lock()

    def run(self):
        """Main loop - updated for continuous listening"""
        self.cap = cv2.VideoCapture(0)

        # Start visual buffer
        self.visual_buffer = RollingVisualBuffer(
            cap=self.cap,
            vjepa_encoder=self.vjepa,
            florence_detector=self.florence
        )
        self.visual_buffer.start()

        # Start continuous listener
        self.listener.start()

        try:
            while True:
                # Process any voice events (non-blocking)
                self._process_voice_events()

                # ... rest of main loop ...
        finally:
            self.listener.stop()
            self.visual_buffer.stop()

    def _process_voice_events(self):
        """Process queued voice events from listener"""
        while True:
            try:
                event = self.voice_event_queue.get_nowait()
            except queue.Empty:
                break

            self._handle_voice_event(event)

    def _handle_voice_event(self, event: VoiceEvent):
        """Route voice events to appropriate handlers"""
        # Get visual context from ~1.5 seconds before speech ended
        # (user was looking at object BEFORE they spoke)
        lookback_time = event.timestamp - 1.5
        visual_context = self.visual_buffer.get_frame_at_time(lookback_time)

        if visual_context is None:
            print("  No visual context available")
            return

        if event.intent == Intent.IDENTIFY:
            self._handle_identify(visual_context)

        elif event.intent == Intent.TEACH:
            if event.extracted_name:
                self._handle_teach(event.extracted_name, visual_context)
            else:
                print(f"  Couldn't parse object name from: '{event.transcript}'")

        elif event.intent == Intent.CORRECT:
            if event.extracted_name:
                self._handle_correct(event.extracted_name)
            else:
                print(f"  Couldn't parse correction from: '{event.transcript}'")

        elif event.intent == Intent.CONFIRM:
            self._handle_confirm()

    def _handle_identify(self, context: BufferedFrame):
        """Handle 'what's this?' queries"""
        # Find focused object from detections
        focus_obj = self.focus.select_focus(
            context.detections,
            context.frame.shape,
            novelty_scorer=self.novelty,
            episode_memory=self.episodes,
            memory_store=self.memory,
            vjepa_encoder=self.vjepa,
            frames=np.array([context.frame])  # Single frame
        )

        if not focus_obj:
            print("  I don't see anything specific to identify")
            return

        # Get embedding for this region
        # Note: Using single frame, may want to grab multiple from buffer
        frames_for_embed = self._get_frames_for_embedding(context.timestamp)
        embedding = self.vjepa.encode_region(frames_for_embed, focus_obj.bbox)

        # Try to match
        match_name, confidence = self.memory.find_match(
            focus_obj.label, embedding, episode_memory=self.episodes
        )

        if match_name:
            print(f"\n  That's your **{match_name}** (confidence: {confidence:.2f})")
            self._set_last_recognition(
                match_name, focus_obj.label, confidence,
                focus_obj.bbox, embedding
            )
        else:
            print(f"\n  I see a {focus_obj.label}, but I don't recognize which specific one")
            print(f"  Say 'this is called [name]' to teach me")

    def _handle_teach(self, object_name: str, context: BufferedFrame):
        """Handle 'this is called X' teaching"""
        # Find focused object
        focus_obj = self.focus.select_focus(
            context.detections,
            context.frame.shape
        )

        if not focus_obj:
            print(f"  I don't see an object to learn as '{object_name}'")
            return

        # Get embedding
        frames_for_embed = self._get_frames_for_embedding(context.timestamp)
        embedding = self.vjepa.encode_region(frames_for_embed, focus_obj.bbox)

        # Learn it
        self.memory.add_object(object_name, focus_obj.label, embedding)
        print(f"\n  Learned: **{object_name}** (category: {focus_obj.label})")

        # Update last recognition
        self._set_last_recognition(
            object_name, focus_obj.label, 1.0,
            focus_obj.bbox, embedding
        )

        # Record episode
        self.episodes.record_episode(
            objects=[{
                'name': object_name,
                'category': focus_obj.label,
                'confidence': 1.0,
                'bbox': focus_obj.bbox,
                'is_focus': True
            }],
            event_type="learning",
            event_detail=f"Learned '{object_name}' via voice teaching"
        )

    def _handle_correct(self, correct_name: str):
        """Handle 'no, that's X' corrections"""
        last = self._get_last_recognition()

        if last is None:
            print("  Nothing to correct - I haven't identified anything recently")
            return

        # Check if correction is recent enough (within 30 seconds)
        if time.time() - last.timestamp > 30:
            print("  Too long since last identification - please show me the object again")
            return

        wrong_name = last.name

        print(f"\n  Correcting: '{wrong_name}' -> '{correct_name}'")

        # Remove bad embedding from wrong object (if it has multiple)
        if wrong_name in self.memory.objects:
            obj = self.memory.objects[wrong_name]
            if len(obj.embeddings) > 1:
                # Find and remove most similar embedding
                max_sim = -1
                max_idx = -1
                for i, stored_emb in enumerate(obj.embeddings):
                    sim = torch.cosine_similarity(
                        last.embedding.cpu().flatten().unsqueeze(0),
                        stored_emb.flatten().unsqueeze(0)
                    ).item()
                    if sim > max_sim:
                        max_sim = sim
                        max_idx = i

                if max_idx >= 0:
                    obj.embeddings.pop(max_idx)
                    print(f"  Removed confusing view from '{wrong_name}'")

        # Add to correct object
        self.memory.add_object(correct_name, last.category, last.embedding)
        print(f"  Added view to '{correct_name}'")

        # Update last recognition
        self._set_last_recognition(
            correct_name, last.category, 1.0,
            last.bbox, last.embedding
        )

        # Record correction episode
        self.episodes.record_episode(
            objects=[{
                'name': correct_name,
                'category': last.category,
                'confidence': 1.0,
                'bbox': last.bbox,
                'is_focus': True
            }],
            event_type="correction",
            event_detail=f"Corrected '{wrong_name}' to '{correct_name}' via voice"
        )

    def _handle_confirm(self):
        """Handle 'yes'/'correct' confirmations"""
        last = self._get_last_recognition()

        if last is None:
            print("  Nothing to confirm")
            return

        if time.time() - last.timestamp > 30:
            print("  Too long since last identification")
            return

        # Strengthen the recognition by adding another embedding
        # (if confidence was low, this helps)
        if last.confidence < 0.95:
            self.memory.add_object(last.name, last.category, last.embedding)
            print(f"\n  Confirmed: **{last.name}** (strengthened memory)")
        else:
            print(f"\n  Confirmed: **{last.name}**")

        # Record confirmation episode
        self.episodes.record_episode(
            objects=[{
                'name': last.name,
                'category': last.category,
                'confidence': last.confidence,
                'bbox': last.bbox,
                'is_focus': True
            }],
            event_type="confirmation",
            event_detail=f"User confirmed identification of '{last.name}'"
        )

    def _get_frames_for_embedding(self, target_time: float) -> np.ndarray:
        """Get multiple frames around target time for V-JEPA embedding"""
        # Get frames from ~0.5s before to target time
        frames = self.visual_buffer.get_frames_in_range(
            target_time - 0.5, target_time
        )

        if len(frames) >= 8:
            # Use frames we have
            return np.array([f.frame for f in frames[-16:]])
        else:
            # Fallback: duplicate the frame we have
            if frames:
                frame = frames[-1].frame
            else:
                latest = self.visual_buffer.get_latest()
                frame = latest.frame if latest else np.zeros((480, 640, 3), dtype=np.uint8)
            return np.array([frame] * 16)
```

### 8. VRAM Budget and Fallbacks

**Current VRAM usage on GTX 1060 (6GB):**
- Florence-2 large: ~2.5GB
- V-JEPA with LoRA: ~1.5GB
- Whisper base: ~500MB
- CUDA overhead: ~500MB
- **Total: ~5GB** (workable)

**If VRAM pressure occurs:**

1. **First fallback**: Use `whisper-tiny` instead of `whisper-base`
   ```python
   self.whisper_model = whisper.load_model("tiny")  # ~150MB vs 500MB
   ```

2. **Second fallback**: Run Whisper on CPU
   ```python
   self.whisper_model = whisper.load_model("base", device="cpu")
   ```
   CPU transcription is still fast enough for short utterances (<3 seconds).

3. **Third fallback**: Reduce visual buffer size
   ```python
   self.visual_buffer = RollingVisualBuffer(..., buffer_seconds=1.5)  # vs 3.0
   ```

**Important**: Whisper and V-JEPA don't run simultaneously in this design:
- V-JEPA runs in background capture loop
- Whisper runs only when speech ends (brief spike)
- This natural interleaving prevents peak VRAM conflicts

## Testing approach

### Phase 1: VAD isolation
```python
# test_vad.py
# Print "speech started" / "speech ended" with timestamps
# Verify silence_duration threshold works in workshop noise
```

### Phase 2: Add Whisper
```python
# Print transcripts after each utterance
# Verify accuracy with workshop background noise
```

### Phase 3: Intent classification
```python
# Print (intent, transcript, extracted_name) tuples
# Test edge cases: "no that's not right" vs "no, that's a screwdriver"
```

### Phase 4: Visual buffer
```python
# Verify frame retrieval by timestamp
# Check thread safety under load
# Verify embedding quality from buffered frames
```

### Phase 5: Full integration
```python
# Test all four intents end-to-end
# Verify temporal binding (lookback gets right frame)
# Test correction/confirmation within time window
```

## Resolved questions

1. **Workshop noise level**: Start with VAD threshold 0.5, tune if needed. May need 0.6-0.7 in noisy conditions.

2. **Response modality**: Print to console for now. Future: audio feedback via TTS.

3. **Threading model**:
   - Main thread: UI, main loop, handler execution
   - Audio thread: VAD, Whisper, intent classification, queues events
   - Visual buffer thread: continuous capture, embedding, detection
   - Communication via `queue.Queue` (thread-safe)

## Future extensions

- **Semantic audio JEPA**: Replace Whisper with direct audio-to-meaning associations
- **Backward attribution**: "Would this be explained by [X]?" - query associative memory for causal hypotheses
- **Audio feedback**: TTS responses ("Got it, that's your Phillips screwdriver")
- **Multi-turn context**: Track conversation state for follow-up questions
