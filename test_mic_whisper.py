import whisper
import pyaudio
import wave
import soundfile as sf

model = whisper.load_model("small")  # Using better model

p = pyaudio.PyAudio()

print("Recording 5 seconds - say something clearly:")
print("Try: 'This is a test of the microphone'")
input("Press ENTER to start...")

stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

print("🎤 RECORDING...")
frames = []
for _ in range(0, int(16000 / 1024 * 5)):
    data = stream.read(1024, exception_on_overflow=False)
    frames.append(data)

print("✓ Done")
stream.stop_stream()
stream.close()
p.terminate()

# Save
audio_path = "mic_test.wav"
wf = wave.open(audio_path, 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(16000)
wf.writeframes(b''.join(frames))
wf.close()

# Transcribe
audio_data, _ = sf.read(audio_path)
if audio_data.dtype != 'float32':
    audio_data = audio_data.astype('float32')

result = model.transcribe(audio_data)

print(f"\nTranscription: '{result['text']}'")
print(f"Language detected: {result.get('language', 'unknown')}")