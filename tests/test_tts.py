import asyncio
from src.core.tts import BernardVoice


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
