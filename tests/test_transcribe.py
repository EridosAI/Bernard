# test_transcribe.py - Phase 2: Test VAD + Whisper transcription
"""
Tests the full audio pipeline: VAD detects speech, Whisper transcribes it.
Verifies that spoken words produce text output through ContinuousListener.

Usage:
    conda activate bernard
    python tests/test_transcribe.py

Speak naturally and verify transcripts appear. Press Ctrl+C to stop.
"""

import sys
import os
import queue
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.continuous_listening import ContinuousListener


def main():
    import whisper

    print("=" * 60)
    print("Transcription Test - Phase 2 (VAD + Whisper)")
    print("=" * 60)

    # Load Whisper
    print("\nLoading Whisper base model...")
    whisper_model = whisper.load_model("base")
    print("Whisper loaded")

    # Event queue
    event_queue = queue.Queue()

    # Create listener
    listener = ContinuousListener(
        whisper_model=whisper_model,
        event_queue=event_queue,
        vad_threshold=0.5,
        silence_duration=0.8
    )

    listener.start()

    print("\n" + "=" * 60)
    print("Listening... Speak to test transcription")
    print("Transcripts will appear below when speech is detected")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    transcript_count = 0

    try:
        while True:
            try:
                event = event_queue.get(timeout=0.5)
                transcript_count += 1
                print(f"  [{event.audio_duration:.1f}s] \"{event.transcript}\"")
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print(f"\n\nStopping... ({transcript_count} transcripts captured)")
        listener.stop()
        print("Done")


if __name__ == "__main__":
    main()
