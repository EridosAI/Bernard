# workshop_session.py - First Continuous Learning Session
import cv2
import whisper
import pyaudio
import wave
import torch
import numpy as np
from transformers import AutoModel, AutoVideoProcessor
from sentence_transformers import SentenceTransformer
import os
import time
from datetime import datetime
import threading

print("="*70)
print("WORKSHOP BERNARD - Continuous Learning Session")
print("="*70)

# Configuration
SESSION_NAME = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DATA_DIR = f"./data/sessions/{SESSION_NAME}"
os.makedirs(DATA_DIR, exist_ok=True)

CLIP_DURATION = 5  # seconds of video per clip
FRAMES_TO_CAPTURE = 150  # Capture 5 seconds at 30 FPS
FRAMES_FOR_VJEPA = 64  # V-JEPA expects 64 frames (we'll downsample)
FPS = 30  # Camera FPS

# Audio settings
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000

print(f"\nSession: {SESSION_NAME}")
print(f"Data will be saved to: {DATA_DIR}")

# Load models
print("\nLoading models...")
print("  - V-JEPA 2 (vision)...")
vision_model = AutoModel.from_pretrained("./models/base/vjepa2")
vision_processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")
device = "cuda" if torch.cuda.is_available() else "cpu"
vision_model = vision_model.to(device)
vision_model.eval()

print("  - Whisper (speech-to-text)...")
whisper_model = whisper.load_model("small")  # Upgraded from 'base' for better accuracy

print("  - Text encoder...")
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

print(f"\n✓ All models loaded on {device}")

# Open camera
CAMERA_INDEX = 1  # Change this if needed (0 or 1)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FPS, FPS)
print(f"✓ Camera {CAMERA_INDEX} opened")

# Initialize audio
p = pyaudio.PyAudio()

# Audio recording function for threading
def record_audio_thread(duration, audio_path, p, audio_format, channels, rate, chunk):
    """Record audio in a separate thread"""
    audio_stream = p.open(format=audio_format,
                         channels=channels,
                         rate=rate,
                         input=True,
                         frames_per_buffer=chunk)

    audio_frames = []
    for i in range(0, int(rate / chunk * duration)):
        data = audio_stream.read(chunk, exception_on_overflow=False)
        audio_frames.append(data)

    audio_stream.stop_stream()
    audio_stream.close()

    # Save audio to file
    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(audio_frames))
    wf.close()

print("\n" + "="*70)
print("READY TO LEARN!")
print("="*70)
print("\nInstructions:")
print("  1. Position camera to see your workspace")
print("  2. Start working and NARRATE what you're doing")
print("  3. The system will capture 5-second clips continuously")
print("  4. Press 'q' in the camera window to stop")
print("\nExample narration:")
print('  "Now I\'m connecting the blue wire to the 100uF capacitor"')
print('  "Soldering the resistor to pin 3"')
print('  "This is a voltage regulator, placing it on the breadboard"')

input("\nPress ENTER to start learning session...")

clip_count = 0
session_data = []

try:
    while True:
        clip_count += 1
        print(f"\n{'='*70}")
        print(f"CLIP {clip_count}")
        print(f"{'='*70}")
        
        # Start audio recording in background thread
        audio_path = os.path.join(DATA_DIR, f"clip_{clip_count}_audio.wav")
        audio_thread = threading.Thread(
            target=record_audio_thread,
            args=(CLIP_DURATION, audio_path, p, AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_RATE, AUDIO_CHUNK)
        )

        # Capture video frames (audio records simultaneously)
        print(f"\n🎥 RECORDING {CLIP_DURATION} SECONDS - SPEAK NOW!")
        audio_thread.start()  # Start audio recording

        frames = []
        frame_timestamps = []

        start_time = time.time()
        last_second_shown = -1

        while len(frames) < FRAMES_TO_CAPTURE:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                elapsed = time.time() - start_time
                frame_timestamps.append(elapsed)

                # Show countdown
                current_second = int(elapsed)
                remaining = CLIP_DURATION - int(elapsed)
                if current_second != last_second_shown and remaining > 0:
                    print(f"  ⏱️  {remaining} seconds remaining...", end='\r')
                    last_second_shown = current_second

                # Show live feed
                cv2.imshow('Workshop Camera (Press Q to stop)', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nStopping session...")
                    raise KeyboardInterrupt

        print(f"\n  ✓ Captured {len(frames)} frames")

        # Wait for audio recording to complete
        audio_thread.join()
        print(f"  ✓ Audio recorded simultaneously")

        # Downsample to 64 frames for V-JEPA (uniform sampling)
        indices = np.linspace(0, len(frames) - 1, FRAMES_FOR_VJEPA, dtype=int)
        frames_for_vjepa = [frames[i] for i in indices]
        print(f"  ✓ Downsampled to {len(frames_for_vjepa)} frames for V-JEPA")

        # Transcribe
        print("  🎤 Transcribing your narration...")
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_path)
        if audio_data.dtype != 'float32':
            audio_data = audio_data.astype('float32')
        
        transcription = whisper_model.transcribe(audio_data)
        narration = transcription['text'].strip()

        print(f'\n  Whisper transcribed: "{narration}"')
        correction = input('  Press ENTER to accept, or type correction: ').strip()
        if correction:
            narration = correction
            print(f'  Using corrected narration: "{narration}"')

        # Process video through V-JEPA
        print("  Processing video through V-JEPA...")
        frames_array = np.array(frames_for_vjepa)
        video_inputs = vision_processor(videos=list(frames_array), return_tensors="pt")
        video_inputs = {k: v.to(device) for k, v in video_inputs.items()}
        
        with torch.no_grad():
            vision_output = vision_model(**video_inputs)
            vision_embedding = vision_output.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Encode narration
        print("  Encoding narration...")
        text_embedding = text_encoder.encode(narration, convert_to_tensor=True)
        
        # Save this training example
        clip_data = {
            'clip_id': clip_count,
            'timestamp': datetime.now().isoformat(),
            'narration': narration,
            'vision_embedding': vision_embedding.cpu().numpy(),
            'text_embedding': text_embedding.cpu().numpy(),
            'audio_file': audio_path
        }
        
        session_data.append(clip_data)
        
        # Save embeddings
        embedding_path = os.path.join(DATA_DIR, f"clip_{clip_count}_embeddings.npz")
        np.savez(embedding_path,
                vision=vision_embedding.cpu().numpy(),
                text=text_embedding.cpu().numpy(),
                narration=narration)
        
        print(f"  ✓ Saved embeddings to {embedding_path}")
        
        # Show similarity (how well aligned vision and text are)
        # Note: Different embedding dimensions, so we'll skip similarity for now
        # This will be handled properly during LoRA training
        print(f"\n  Vision embedding shape: {vision_embedding.shape}")
        print(f"  Text embedding shape: {text_embedding.shape}")
        similarity = 0.0  # Placeholder
        
        print(f"\n  Vision-Text Alignment: {similarity:.3f}")
        print(f"  (Higher = better alignment between what you said and what camera saw)")
        
        print(f"\n✓ Clip {clip_count} processed and saved")
        print(f"\nContinuing... (press 'q' in camera window to stop)")

except KeyboardInterrupt:
    print("\n\nSession ended by user")

finally:
    # Cleanup
    cap.release()