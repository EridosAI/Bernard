# active_jarvis_v4.py - Production-ready active learning
import cv2
import torch
import torch.nn as nn
import whisper
import pyaudio
import wave
import soundfile as sf
from transformers import AutoModel, AutoVideoProcessor
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import re
# At startup - load saved objects
import json

def load_knowledge():
    try:
        with open('known_objects.json', 'r') as f:
            data = json.load(f)
            for name in data:
                text_emb = text_encoder.encode(name, convert_to_tensor=True).to(device)
                known_objects[name] = text_emb
        print(f"✓ Loaded {len(known_objects)} objects from last session")
    except:
        print("Starting fresh session")

def save_knowledge():
    with open('known_objects.json', 'w') as f:
        json.dump(list(known_objects.keys()), f, indent=2)
    print(f"✓ Saved {len(known_objects)} objects")

# Call load_knowledge() at startup
# Call save_knowledge() before exiting

print("="*70)
print("WORKSHOP JARVIS - Active Learning System")
print("="*70)

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModel.from_pretrained("./models/base/vjepa2")
vision_model = PeftModel.from_pretrained(
    base_model,
    "./models/adapters/workshop_lora_20260117_041817"
).to(device).eval()

processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
whisper_model = whisper.load_model("small")

class AlignmentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_proj = nn.Linear(1024, 512)
        self.text_proj = nn.Linear(384, 512)
        
    def forward(self, vision_emb, text_emb):
        vision_proj = self.vision_proj(vision_emb)
        text_proj = self.text_proj(text_emb)
        return vision_proj, text_proj

alignment_head = AlignmentHead().to(device)
alignment_head.load_state_dict(
    torch.load("./models/adapters/workshop_lora_20260117_041817/alignment_head.pt",
               weights_only=True)
)
alignment_head.eval()

print("✓ Models loaded")

# Configuration
known_objects = {}
previous_scene = None
CONFIDENCE_THRESHOLD = 0.15
CHANGE_THRESHOLD = 0.93

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
    cv2.destroyAllWindows()
    
    frames_array = np.array(frames)
    inputs = processor(videos=list(frames_array), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = vision_model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1)
    
    return embedding

def scene_changed(current_emb, previous_emb):
    """Check if scene changed significantly"""
    if previous_emb is None:
        return True
    
    sim = torch.cosine_similarity(current_emb, previous_emb).item()
    changed = sim < CHANGE_THRESHOLD
    
    if changed:
        print(f"  ✓ Change detected (similarity: {sim:.3f})")
    
    return changed

def find_match(vision_emb):
    """Search known objects"""
    if not known_objects:
        return None, 0
    
    best_match = None
    best_confidence = 0
    
    with torch.no_grad():
        for name, text_emb in known_objects.items():
            vision_proj, text_proj = alignment_head(vision_emb, text_emb.unsqueeze(0))
            
            vision_proj = vision_proj.squeeze(1) if vision_proj.dim() > 2 else vision_proj
            text_proj = text_proj.squeeze(1) if text_proj.dim() > 2 else text_proj
            
            vision_norm = vision_proj / vision_proj.norm()
            text_norm = text_proj / text_proj.norm()
            
            sim = torch.cosine_similarity(vision_norm, text_norm).item()
            
            if sim > best_confidence:
                best_confidence = sim
                best_match = name
    
    return best_match, best_confidence

def ask_question():
    """Ask what the object is"""
    print(f"\n🤖 JARVIS: What is that object?")
    input("   Press ENTER when ready to answer... ")
    
    # Record
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    print("   🎤 Speak now!(5 seconds)")
    frames = []
    for _ in range(0, int(16000 / 1024 * 5)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Transcribe
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
    return result['text'].strip()

def clean_response(answer):
    """Extract clean object name"""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove common phrases
    remove_phrases = [
        "that's a ", "that is a ", "that's an ", "that is an ",
        "it's a ", "it is a ", "it's an ", "it is an ",
        "this is a ", "this is an ", "this is ",
        "that's ", "it's ", "this is", "the ", "a ", "an "
    ]
    
    for phrase in remove_phrases:
        answer = answer.replace(phrase, "")
    
    # Remove punctuation
    answer = re.sub(r'[.,!?;:]', '', answer)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer.strip()

def is_valid_object_name(name):
    """Check if this is a real object name"""
    if len(name) < 3:
        return False
    
    # Filter garbage
    garbage = ['um', 'uh', 'the', 'what', 'where', 'how', 'why', 'on a', 'in a']
    if name in garbage:
        return False
    
    # Must have at least one letter
    if not any(c.isalpha() for c in name):
        return False
    
    return True

# Main loop
print("\nActive Learning Mode")
print("Show me objects and I'll learn what they are!\n")

while True:
    try:
        print("-" * 50)
        
        # Capture scene
        vision_emb = capture_scene()
        
        # Check if scene changed
        if not scene_changed(vision_emb, previous_scene):
            previous_scene = vision_emb
            time.sleep(5)
            continue
        
        # Try to identify
        match, confidence = find_match(vision_emb)
        
        if match and confidence > CONFIDENCE_THRESHOLD:
            print(f"✓ I recognize this: **{match}** (conf: {confidence:.2f})")
        
        elif match and confidence > 0.08:  # Uncertain but has a guess
            print(f"⚠ I think this might be '{match}' but I'm only {confidence:.0%} confident.")
            confirm = input("   Is that correct? (y/n): ").strip().lower()
            
            if confirm == 'y':
                print(f"   ✓ Good! Reinforcing memory of '{match}'")
            else:
                print(f"   Okay, let me learn what it really is...")
                answer = ask_question()
                print(f"   You said: '{answer}'")
                
                cleaned = clean_response(answer)
                if is_valid_object_name(cleaned):
                    text_emb = text_encoder.encode(cleaned, convert_to_tensor=True).to(device)
                    known_objects[cleaned] = text_emb
                    print(f"   ✓ Learned: **{cleaned}**")
                else:
                    print(f"   ⚠ Skipped (invalid name)")
        
        else:
            # Completely unknown - ask what it is
            answer = ask_question()
            print(f"   You said: '{answer}'")
            
            cleaned = clean_response(answer)
            if is_valid_object_name(cleaned):
                text_emb = text_encoder.encode(cleaned, convert_to_tensor=True).to(device)
                known_objects[cleaned] = text_emb
                print(f"   ✓ Learned: **{cleaned}**")
            else:
                print(f"   ⚠ Skipped (invalid name)")
        
        previous_scene = vision_emb
        time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\nSession ended")
        print(f"\nLearned {len(known_objects)} objects:")
        for i, obj in enumerate(known_objects.keys(), 1):
            print(f"  {i}. {obj}")
        break
    
    except Exception as e:
        print(f"Error: {e}")
        continue