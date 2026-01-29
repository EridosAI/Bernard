# Continuous Listening Spec

## Overview

Always-on voice interaction without wake words. Natural workshop conversation flows directly to the system.

**Pipeline:** Mic → VAD (Silero) → Whisper → Intent Classification → Event Dispatch

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Audio Stream│────▶│ Silero VAD  │────▶│   Whisper   │────▶│   Intent    │
│  (16kHz)    │     │ (CPU, 30ms) │     │ (GPU, base) │     │  Classify   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                                       │
                    speech start/end                        VoiceEvent
                           │                                       │
                           ▼                                       ▼
                    ┌─────────────┐                         ┌─────────────┐
                    │Audio Buffer │                         │  Callback   │
                    │  (growing)  │                         │  Handler    │
                    └─────────────┘                         └─────────────┘
```

## Dependencies

```bash
pip install silero-vad sounddevice --break-system-packages
# Whisper should already be available
```

## Data Structures

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np

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
    timestamp: float              # when speech ended (time.time())
    audio_duration: float         # how long the utterance was
    extracted_object: Optional[str] = None  # parsed object name for TEACH/CORRECT
```

## ContinuousListener Class

```python
class ContinuousListener:
    """
    Always-on voice pipeline: VAD → Whisper → Intent classification
    
    Usage:
        def handle_voice(event: VoiceEvent):
            print(f"{event.intent}: {event.transcript}")
        
        listener = ContinuousListener(event_callback=handle_voice)
        listener.start()
        # ... later ...
        listener.stop()
    """
    
    def __init__(
        self,
        event_callback,               # Called with VoiceEvent when intent != IGNORE
        whisper_model: str = "base",  # "tiny", "base", "small" — GTX 1060 constraint
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,   # Silero confidence threshold
        silence_duration: float = 0.8 # Seconds of silence to end utterance
    ):
        self.callback = event_callback
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        
        # Load models
        self._load_vad()
        self._load_whisper(whisper_model)
        
        # State
        self.running = False
        self.audio_thread = None
    
    def start(self):
        """Start listening in background thread"""
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
    
    def stop(self):
        """Stop listening, cleanup resources"""
        self.running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
```

## Internal Methods

### `_load_vad()`
```python
def _load_vad(self):
    self.vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    self.vad_model.eval()
```

### `_load_whisper(model_name)`
```python
def _load_whisper(self, model_name: str):
    import whisper
    self.whisper_model = whisper.load_model(model_name)
```

### `_audio_loop()`
Main loop — capture audio chunks, run VAD, buffer during speech, process on speech end.

```python
def _audio_loop(self):
    import sounddevice as sd
    
    chunk_size = 512  # ~32ms at 16kHz
    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    speech_start_time = None
    
    with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
        while self.running:
            chunk, _ = stream.read(chunk_size)
            chunk = chunk.flatten()
            
            # Run VAD
            speech_prob = self._get_speech_prob(chunk)
            
            if speech_prob > self.vad_threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_start_time = time.time()
                    audio_buffer = []
                audio_buffer.append(chunk)
                silence_samples = 0
            else:
                if is_speaking:
                    audio_buffer.append(chunk)
                    silence_samples += len(chunk)
                    
                    # Check if silence duration exceeded
                    if silence_samples / self.sample_rate > self.silence_duration:
                        # Speech ended — process
                        self._process_utterance(
                            np.concatenate(audio_buffer),
                            speech_start_time
                        )
                        is_speaking = False
                        audio_buffer = []
```

### `_get_speech_prob(chunk)`
```python
def _get_speech_prob(self, chunk: np.ndarray) -> float:
    tensor = torch.from_numpy(chunk).float()
    return self.vad_model(tensor, self.sample_rate).item()
```

### `_process_utterance(audio, start_time)`
```python
def _process_utterance(self, audio: np.ndarray, start_time: float):
    end_time = time.time()
    duration = end_time - start_time
    
    # Transcribe
    result = self.whisper_model.transcribe(
        audio,
        language="en",
        fp16=False  # GTX 1060 compatibility
    )
    transcript = result["text"].strip()
    
    if not transcript:
        return
    
    # Classify intent
    intent, extracted = self._classify_intent(transcript)
    
    if intent != Intent.IGNORE:
        event = VoiceEvent(
            intent=intent,
            transcript=transcript,
            timestamp=end_time,
            audio_duration=duration,
            extracted_object=extracted
        )
        self.callback(event)
```

### `_classify_intent(transcript) -> Tuple[Intent, Optional[str]]`

```python
def _classify_intent(self, transcript: str) -> Tuple[Intent, Optional[str]]:
    text = transcript.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # strip punctuation
    
    # CONFIRM — exact matches
    confirm_phrases = {"yes", "yeah", "yep", "right", "correct", "thats right", "that's right"}
    if text in confirm_phrases:
        return Intent.CONFIRM, None
    
    # IDENTIFY — questions about objects
    if text.startswith(("whats", "what's", "what is")):
        if "this" in text or "that" in text:
            return Intent.IDENTIFY, None
    
    # CORRECT — corrections
    correct_prefixes = ["no ", "actually ", "thats not", "that's not", "its not", "it's not"]
    for prefix in correct_prefixes:
        if text.startswith(prefix):
            remainder = text[len(prefix):].strip()
            extracted = self._extract_object_name(remainder)
            return Intent.CORRECT, extracted
    
    # TEACH — teaching new objects
    teach_prefixes = [
        "this is called a ", "this is called ", 
        "this is a ", "this is ", 
        "thats a ", "that's a ", "that is a ",
        "its called ", "it's called "
    ]
    for prefix in teach_prefixes:
        if text.startswith(prefix):
            extracted = text[len(prefix):].strip()
            return Intent.TEACH, extracted if extracted else None
    
    return Intent.IGNORE, None

def _extract_object_name(self, text: str) -> Optional[str]:
    """Extract object name from correction/teach phrase"""
    # Handle "that's a X" or "it's a X" patterns in remainder
    for prefix in ["thats a ", "that's a ", "its a ", "it's a ", "a "]:
        if text.startswith(prefix):
            return text[len(prefix):].strip() or None
    return text.strip() or None
```

## Integration with Main System

In `bernard.py` (or main system file), add handler:

```python
from continuous_listening import ContinuousListener, VoiceEvent, Intent

class BernardSystem:
    def __init__(self, ...):
        # ... existing init ...
        
        # Voice listener
        self.listener = ContinuousListener(
            event_callback=self._handle_voice_event
        )
    
    def _handle_voice_event(self, event: VoiceEvent):
        """Route voice events to appropriate handlers"""
        
        if event.intent == Intent.IDENTIFY:
            # Trigger identification of current focus object
            self._identify_current_object()
        
        elif event.intent == Intent.TEACH:
            # Learn new object name
            self._teach_object(event.extracted_object, event.timestamp)
        
        elif event.intent == Intent.CORRECT:
            # Correct most recent identification
            self._correct_object(event.extracted_object, event.timestamp)
        
        elif event.intent == Intent.CONFIRM:
            # Positive reinforcement for recent identification
            self._confirm_identification()
    
    def start(self):
        # ... existing start ...
        self.listener.start()
    
    def stop(self):
        self.listener.stop()
        # ... existing stop ...
```

## Temporal Binding

Voice events need binding to visual context. The `timestamp` field allows correlation with recent frames/embeddings.

In main system, maintain a rolling buffer:
```python
from collections import deque

# 3 seconds at 30fps
self.visual_buffer = deque(maxlen=90)

# On each frame:
self.visual_buffer.append({
    'timestamp': time.time(),
    'frame': frame,
    'embedding': embedding,
    'detected_objects': objects
})

# When voice event arrives, find closest visual context:
def _get_visual_context(self, voice_timestamp: float, lookback: float = 3.0):
    """Get visual context from around voice event time"""
    candidates = [
        v for v in self.visual_buffer 
        if voice_timestamp - lookback <= v['timestamp'] <= voice_timestamp
    ]
    return candidates[-1] if candidates else None
```

## Testing Phases

### Phase 1: VAD isolation
```python
# test_vad.py
listener = ContinuousListener(lambda e: None)
listener._load_vad()

# Manual test: speak, verify detection
with sd.InputStream(...) as stream:
    while True:
        chunk = stream.read(512)
        prob = listener._get_speech_prob(chunk)
        if prob > 0.5:
            print("SPEECH", end="", flush=True)
        else:
            print(".", end="", flush=True)
```

### Phase 2: VAD + Whisper
```python
# test_transcribe.py
def print_transcript(event):
    print(f"[{event.audio_duration:.1f}s] {event.transcript}")

listener = ContinuousListener(event_callback=print_transcript)
listener.start()
input("Press Enter to stop...")
listener.stop()
```

### Phase 3: Full intent classification
```python
# test_intent.py
def print_event(event):
    print(f"{event.intent.value}: '{event.transcript}'")
    if event.extracted_object:
        print(f"  -> object: {event.extracted_object}")

listener = ContinuousListener(event_callback=print_event)
listener.start()
input("Press Enter to stop...")
listener.stop()
```

## Hardware Notes

- **Workshop PC**: GTX 1060 (6GB VRAM)
- VAD runs on CPU — no GPU pressure
- Whisper `base` model uses ~1GB VRAM
- V-JEPA and Whisper don't run simultaneously (Whisper only on utterance end)
- If VRAM pressure occurs: try `whisper-tiny` or `--device cpu`

## Open Questions for Implementation

1. **Workshop noise** — May need to tune `vad_threshold` based on ambient noise level
2. **Audio feedback** — Currently silent operation; could add TTS confirmation
3. **Thread coordination** — Audio thread is daemon; verify clean shutdown with main capture thread

## File Structure

```
src/
  continuous_listening.py   # ContinuousListener class
  test_vad.py               # Phase 1 test
  test_transcribe.py        # Phase 2 test  
  test_intent.py            # Phase 3 test
```
