# TTS Integration Spec

## Overview

Give Bernard a voice. When the system identifies an object, confirms learning, or responds to a query, it should speak the response aloud.

**Priority:** Best experience with minimal compute overhead. The GTX 1060 is already running V-JEPA, Whisper, and Florence-2.

## Recommended: edge-tts

Microsoft Edge's neural TTS. Free, high quality, zero local GPU usage.

**Why this:**
- Neural voice quality (not robotic)
- Zero VRAM — runs via Microsoft's cloud
- Australian voices available (you're in Perth)
- Async — won't block the main loop
- Free, no API key needed

**Tradeoff:** Requires internet. Fine for workshop with WiFi.

### Installation

```bash
pip install edge-tts pygame --break-system-packages
```

### Implementation

```python
# src/tts.py

import edge_tts
import asyncio
import tempfile
import os
from pathlib import Path

# pygame for audio playback (lighter than alternatives)
import pygame
pygame.mixer.init()

class BernardVoice:
    """
    Text-to-speech for Bernard using Edge TTS.
    
    Usage:
        voice = BernardVoice()
        voice.speak("I see a multimeter")
        
        # Or async:
        await voice.speak_async("I see a multimeter")
    """
    
    def __init__(
        self,
        voice: str = "en-AU-WilliamNeural",  # Australian male
        rate: str = "+0%",                    # Speed adjustment
        cache_dir: str = None                 # Cache repeated phrases
    ):
        self.voice = voice
        self.rate = rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
        
        self._speaking = False
    
    async def speak_async(self, text: str) -> None:
        """Generate and play speech asynchronously"""
        if self._speaking:
            return  # Don't overlap speech
        
        self._speaking = True
        try:
            # Check cache first
            audio_path = self._get_cached(text)
            
            if not audio_path:
                # Generate new audio
                audio_path = await self._generate(text)
            
            # Play audio
            self._play(audio_path)
            
        finally:
            self._speaking = False
    
    def speak(self, text: str) -> None:
        """Synchronous wrapper for speak_async"""
        asyncio.run(self.speak_async(text))
    
    async def _generate(self, text: str) -> str:
        """Generate audio file from text"""
        communicate = edge_tts.Communicate(
            text, 
            self.voice,
            rate=self.rate
        )
        
        # Use cache dir or temp file
        if self.cache_dir:
            # Hash the text for cache filename
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            audio_path = self.cache_dir / f"{text_hash}.mp3"
            if not audio_path.exists():
                await communicate.save(str(audio_path))
        else:
            # Temp file
            fd, audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            await communicate.save(audio_path)
        
        return str(audio_path)
    
    def _get_cached(self, text: str) -> str | None:
        """Check if we have cached audio for this text"""
        if not self.cache_dir:
            return None
        
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        audio_path = self.cache_dir / f"{text_hash}.mp3"
        
        return str(audio_path) if audio_path.exists() else None
    
    def _play(self, audio_path: str) -> None:
        """Play audio file"""
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    
    def stop(self) -> None:
        """Stop current playback"""
        pygame.mixer.music.stop()
        self._speaking = False


# Available Australian voices:
# - en-AU-WilliamNeural (male, recommended)
# - en-AU-NatashaNeural (female)
# 
# Other good options:
# - en-GB-RyanNeural (British male)
# - en-US-GuyNeural (American male)
```

### Integration with Arnold

```python
# In arnold.py or wherever responses are generated

from tts import BernardVoice

class ArnoldSystem:
    def __init__(self, ...):
        # ... existing init ...
        
        self.voice = BernardVoice(
            voice="en-AU-WilliamNeural",
            cache_dir="./tts_cache"  # Cache common phrases
        )
    
    def _handle_voice_event(self, event: VoiceEvent):
        if event.intent == Intent.IDENTIFY:
            result = self._identify_current_object()
            if result:
                self.voice.speak(f"That's a {result.name}")
            else:
                self.voice.speak("I'm not sure what that is")
        
        elif event.intent == Intent.TEACH:
            obj_name = event.extracted_object
            self._teach_object(obj_name, event.timestamp)
            self.voice.speak(f"Got it. Learning {obj_name}")
        
        elif event.intent == Intent.CORRECT:
            obj_name = event.extracted_object
            self._correct_object(obj_name, event.timestamp)
            self.voice.speak(f"Thanks. Correcting to {obj_name}")
        
        elif event.intent == Intent.CONFIRM:
            self._confirm_identification()
            self.voice.speak("Good")  # Keep confirmations short
```

### Common Phrases to Pre-cache

```python
# Pre-generate common responses on startup
COMMON_PHRASES = [
    "Got it",
    "Good", 
    "Thanks",
    "I'm not sure what that is",
    "I don't recognise that",
    "Learning",
    "Correcting",
    "I see",
]

async def warm_cache(voice: BernardVoice):
    for phrase in COMMON_PHRASES:
        await voice._generate(phrase)
```

## Alternative: Piper (Fully Offline)

If you want to run without internet, Piper is the best local option. Fast, CPU-based, decent quality.

```bash
pip install piper-tts --break-system-packages
```

```python
from piper import PiperVoice

voice = PiperVoice.load("en_GB-alan-medium")
audio = voice.synthesize("Hello from Bernard")
# Write to file or play directly
```

**Tradeoff:** Slightly lower quality than Edge, needs to download voice models (~100MB each).

## Test Script

```python
# tests/test_tts.py

import asyncio
from src.tts import BernardVoice

async def test_voice():
    voice = BernardVoice()
    
    print("Testing TTS...")
    
    # Test basic speech
    await voice.speak_async("Hello, I'm Bernard. I can see your workshop.")
    
    # Test quick responses
    await voice.speak_async("Got it. Learning multimeter.")
    
    # Test longer response
    await voice.speak_async("That looks like a Phillips head screwdriver. I've seen one of those before.")
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(test_voice())
```

## Performance Notes

- **Latency:** ~200-500ms for short phrases (network round-trip to Edge)
- **Caching:** Eliminates latency for repeated phrases
- **Overlap:** Current implementation blocks until speech finishes — appropriate for workshop interaction
- **VRAM:** Zero — all compute is cloud-side

## Future: Full-Duplex

Eventually you mentioned PersonaPlex-7B for overlapping conversation. When that happens, the TTS layer stays the same — you'd just need to handle interruption (stop current playback when user starts speaking).

For now, the simple blocking model is fine for workshop interaction.
