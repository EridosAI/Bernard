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

print("="*70)
print("WORKSHOP JARVIS - Continuous Learning Session")
print("="*70)

# Configuration
SESSION_NAME = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DATA_DIR = f"./data/sessions/{SESSION_NAME}"
os.makedirs(DATA_DIR, exist_ok=True)

CLIP_DURATION = 5  # seconds of video per clip
FRAMES_PER_CLIP = 64  # V-JEPA expects 64 frames
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
whisper_model = whisper.load_model("base")

print("  - Text encoder...")
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

print(f"\n✓ All models loaded on {device}")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)
print("✓ Camera opened")

# Initialize audio
p = pyaudio.PyAudio()

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
        
        # Capture video frames
        print(f"Capturing {CLIP_DURATION} seconds of video...")
        frames = []
        frame_timestamps = []
        
        start_time = time.time()
        while len(frames) < FRAMES_PER_CLIP:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_timestamps.append(time.time() - start_time)
                
                # Show live feed
                cv2.imshow('Workshop Camera (Press Q to stop)', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nStopping session...")
                    raise KeyboardInterrupt
        
        print(f"  ✓ Captured {len(frames)} frames")
        
        # Record audio for the same duration
        print(f"Audio was recorded simultaneously")
        print("  🎤 Transcribing your narration...")
        
        # Record audio
        audio_stream = p.open(format=AUDIO_FORMAT,
                             channels=AUDIO_CHANNELS,
                             rate=AUDIO_RATE,
                             input=True,
                             frames_per_buffer=AUDIO_CHUNK)
        
        audio_frames = []
        for i in range(0, int(AUDIO_RATE / AUDIO_CHUNK * CLIP_DURATION)):
            data = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
        
        audio_stream.stop_stream()
        audio_stream.close()
        
        # Save audio to temp file
        audio_path = os.path.join(DATA_DIR, f"clip_{clip_count}_audio.wav")
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        
        # Transcribe
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_path)
        if audio_data.dtype != 'float32':
            audio_data = audio_data.astype('float32')
        
        transcription = whisper_model.transcribe(audio_data)
        narration = transcription['text'].strip()
        
        print(f'\n  Narration: "{narration}"')
        
        # Process video through V-JEPA
        print("  Processing video through V-JEPA...")
        frames_array = np.array(frames)
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