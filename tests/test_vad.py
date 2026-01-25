# test_vad.py - Phase 1: Test VAD in isolation
"""
Tests Silero VAD speech detection without Whisper.
Use this to verify VAD threshold works in workshop environment.
"""

import time
import numpy as np
import torch
import sounddevice as sd

# VAD configuration
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz (minimum for Silero VAD)
VAD_THRESHOLD = 0.5  # Adjust based on workshop noise
SILENCE_DURATION = 0.8  # Seconds of silence to end utterance


def main():
    print("=" * 60)
    print("VAD Test - Phase 1")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Sample rate: {SAMPLE_RATE}")
    print(f"  VAD threshold: {VAD_THRESHOLD}")
    print(f"  Silence duration: {SILENCE_DURATION}s")

    # Load Silero VAD
    print("\nLoading Silero VAD...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    vad_model.eval()
    print("VAD loaded")

    # State tracking
    is_speaking = False
    silence_chunks = 0
    speech_start_time = None
    chunk_duration = CHUNK_SAMPLES / SAMPLE_RATE
    silence_chunks_threshold = int(SILENCE_DURATION / chunk_duration)

    print("\n" + "=" * 60)
    print("Listening... Speak to test VAD detection")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_chunks, speech_start_time

        if status:
            print(f"  Audio status: {status}")

        # Convert to format VAD expects
        audio_chunk = indata[:, 0].copy()
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Run VAD
        try:
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
        except Exception as e:
            print(f"  VAD error: {e}")
            return

        is_speech = speech_prob > VAD_THRESHOLD

        if is_speech:
            if not is_speaking:
                # Speech started
                is_speaking = True
                silence_chunks = 0
                speech_start_time = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] SPEECH STARTED (prob: {speech_prob:.2f})")

            silence_chunks = 0

        else:
            if is_speaking:
                silence_chunks += 1

                if silence_chunks >= silence_chunks_threshold:
                    # Speech ended
                    duration = time.time() - speech_start_time
                    print(f"[{time.strftime('%H:%M:%S')}] SPEECH ENDED (duration: {duration:.2f}s)")
                    is_speaking = False
                    silence_chunks = 0
                    speech_start_time = None

                    # Reset VAD state
                    vad_model.reset_states()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback
        ):
            print("Audio stream started\n")
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
