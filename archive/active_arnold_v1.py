# active_arnold_v1.py - Simplified active learning prototype
import cv2
import torch
import whisper
import pyaudio
import wave
import soundfile as sf
from transformers import AutoModel, AutoVideoProcessor
from sentence_transformers import SentenceTransformer
import numpy as np
import time

print("="*70)
print("ACTIVE ARNOLD - Prototype v1")
print("="*70)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
vision_model = AutoModel.from_pretrained("./models/base/vjepa2").to(device).eval()
processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
whisper_model = whisper.load_model("base")

# Knowledge base
known_objects = {}
CONFIDENCE_THRESHOLD = 0.95

def capture_scene():
    """Capture current scene"""
    cap = cv2.VideoCapture(0)
    frames = []
    
    for _ in range(64):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    # Process through V-JEPA
    frames_array = np.array(frames)
    inputs = processor(videos=list(frames_array), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = vision_model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1)
    
    return embedding

def extract_color(vision_emb):
    """Try to detect color"""
    colors = ['red', 'blue', 'green', 'white', 'black', 'yellow']
    best_color = None
    best_score = 0
    
    # Flatten vision embedding to 1D for comparison
    vision_flat = vision_emb.flatten()
    
    for color in colors:
        text_emb = text_encoder.encode(color, convert_to_tensor=True).to(device)
        
        # Project both to same dimension by taking dot product with normalization
        # Simple similarity: just use first N dimensions
        min_dim = min(vision_flat.shape[0], text_emb.shape[0])
        
        vision_truncated = vision_flat[:min_dim]
        text_truncated = text_emb[:min_dim]
        
        # Normalize and compare
        vision_norm = vision_truncated / vision_truncated.norm()
        text_norm = text_truncated / text_truncated.norm()
        
        score = torch.dot(vision_norm, text_norm).item()
        
        if score > best_score:
            best_score = score
            best_color = color
    
    if best_score > 0.3:  # Threshold
        return best_color
    return None
def find_match(vision_emb):
    """Search known objects"""
    best_match = None
    best_confidence = 0
    
    vision_flat = vision_emb.flatten()
    
    for name, stored_emb in known_objects.items():
        stored_flat = stored_emb.flatten()
        
        # Compare same dimensions
        min_dim = min(vision_flat.shape[0], stored_flat.shape[0])
        sim = torch.cosine_similarity(
            vision_flat[:min_dim].unsqueeze(0),
            stored_flat[:min_dim].unsqueeze(0)
        ).item()
        
        if sim > best_confidence:
            best_confidence = sim
            best_match = name
    
    return best_match, best_confidence

def ask_question(question):
    """Ask via text, get voice answer"""
    print(f"\n🤖 ARNOLD: {question}")
    
    # Record answer
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    print("   🎤 Listening... (speak now)")
    frames = []
    for _ in range(0, int(16000 / 1024 * 3)):  # 3 seconds
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save and transcribe
    audio_path = "temp_answer.wav"
    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    audio_data, _ = sf.read(audio_path)
    if audio_data.dtype != 'float32':
        audio_data = audio_data.astype('float32')
    
    result = whisper_model.transcribe(audio_data)
    answer = result['text'].strip()
    
    print(f"   👤 You: {answer}")
    return answer

# Main loop
print("\nStarting active monitoring...")
print("Arnold will watch and ask questions when uncertain\n")

try:
    while True:
        print("\n" + "-"*70)
        print("Observing workspace...")
        
        # Capture scene
        vision_emb = capture_scene()
        
        # Try to identify
        match, confidence = find_match(vision_emb)
        
        if match and confidence > CONFIDENCE_THRESHOLD:
            print(f"✓ I see a {match} (confidence: {confidence:.2f})")
        
        else:
            # Try to describe what we DO know
            color = extract_color(vision_emb)
            
            if color:
                question = f"I see a {color} object. What is that?"
            else:
                question = "What is that object?"
            
            answer = ask_question(question)
            
            if answer and len(answer) > 2:
                # Learn it
                known_objects[answer] = vision_emb
                print(f"   ✓ Learned: '{answer}'")
            else:
                print("   (Didn't catch that)")
        
        # Wait before next check
        time.sleep(10)

except KeyboardInterrupt:
    print("\n\nStopping Arnold...")
    print(f"Learned {len(known_objects)} objects: {list(known_objects.keys())}")