import edge_tts
import asyncio
import tempfile
import os
import hashlib
from pathlib import Path

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

    # Available Australian voices:
    # - en-AU-WilliamNeural (male, recommended)
    # - en-AU-NatashaNeural (female)
    #
    # Other good options:
    # - en-GB-RyanNeural (British male)
    # - en-US-GuyNeural (American male)

    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+0%",
        cache_dir: str = None
    ):
        self.voice = voice
        self.rate = rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

        self._speaking = False

    async def speak_async(self, text: str) -> None:
        """Generate and play speech asynchronously."""
        if self._speaking:
            return  # Don't overlap speech

        self._speaking = True
        try:
            # Check cache first
            audio_path = self._get_cached(text)

            if not audio_path:
                audio_path = await self._generate(text)

            self._play(audio_path)

        finally:
            self._speaking = False

    def speak(self, text: str) -> None:
        """Synchronous wrapper for speak_async."""
        asyncio.run(self.speak_async(text))

    async def _generate(self, text: str) -> str:
        """Generate audio file from text."""
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate
        )

        if self.cache_dir:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            audio_path = self.cache_dir / f"{text_hash}.mp3"
            if not audio_path.exists():
                await communicate.save(str(audio_path))
        else:
            fd, audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            await communicate.save(audio_path)

        return str(audio_path)

    def _get_cached(self, text: str) -> str | None:
        """Check if we have cached audio for this text."""
        if not self.cache_dir:
            return None

        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        audio_path = self.cache_dir / f"{text_hash}.mp3"

        return str(audio_path) if audio_path.exists() else None

    def _play(self, audio_path: str) -> None:
        """Play audio file."""
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

    def stop(self) -> None:
        """Stop current playback."""
        pygame.mixer.music.stop()
        self._speaking = False


# Common phrases to pre-cache on startup
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
    """Pre-generate common responses to eliminate latency."""
    for phrase in COMMON_PHRASES:
        await voice._generate(phrase)
