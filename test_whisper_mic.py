# test_whisper_mic.py
import whisper
import pyaudio
import wave
import os
import time

print("Testing microphone + Whisper speech-to-text...\n")

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("✓ Whisper loaded\n")

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

print("="*60)
print("MICROPHONE TEST")
print("="*60)
print(f"\nRecording for {RECORD_SECONDS} seconds...")
print("Say something like: 'Now I'm soldering the blue wire to the capacitor'")
print("\nRecording starts in 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("\n🎤 RECORDING NOW - SPEAK!\n")

# Initialize PyAudio
p = pyaudio.PyAudio()

try:
    # Open microphone stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    # Record audio
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Progress indicator
        if i % 10 == 0:
            print(".", end="", flush=True)
    
    print("\n\n✓ Recording complete!")
    
    # Stop and close stream
    stream.stop_stream()
    stream.close()
    
    # Save to WAV file
    audio_path = "test_recording.wav"
    
    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"✓ Audio saved to: {audio_path}")
    print(f"\nTranscribing with Whisper...")
    
    # Transcribe with Whisper
    # Load audio directly with soundfile (no FFmpeg needed)
    import soundfile as sf
    audio_data, sample_rate = sf.read(audio_path)
    
    # Convert to float32 if needed
    if audio_data.dtype != 'float32':
        audio_data = audio_data.astype('float32')
    
    result = model.transcribe(audio_data)
    
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT:")
    print("="*60)
    print(f"\n{result['text']}\n")
    print("="*60)
    
    # Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print("\n✓ Temp file cleaned up")
    
    print("\n✓ Microphone + Whisper working perfectly!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    p.terminate()
    print("\n" + "="*60)
    print("Test complete!")