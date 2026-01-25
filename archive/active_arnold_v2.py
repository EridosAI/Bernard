# active_arnold_v2.py - With trained alignment head
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

print("="*70)
print("ACTIVE ARNOLD - v2 (with trained model)")
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
whisper_model = whisper.load_model("base")

# Load alignment head
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

print("✓ Loaded trained model")

# Knowledge base
known_objects = {}
CONFIDENCE_THRESHOLD = 0.6  # Can be lower with alignment head

def capture_scene():
    """Capture current scene"""
    cap = cv2.VideoCapture(0)
    frames = []
    
    print("  📹 Capturing...")
    for _ in range(64):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Process through V-JEPA
    frames_array = np.array(frames)
    inputs = processor(videos=list(frames_array), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = vision_model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1)
    
    return embedding

def find_match(vision_emb):
    """Search known objects using alignment head"""
    if not known_objects:
        return None, 0
    
    best_match = None
    best_confidence = 0
    
    with torch.no_grad():
        for name, text_emb in known_objects.items():
            # Project through alignment head
            vision_proj, text_proj = alignment_head(vision_emb, text_emb.unsqueeze(0))
            
            # Normalize
            vision_proj = vision_proj.squeeze(1) if vision_proj.dim() > 2 else vision_proj
            text_proj = text_proj.squeeze(1) if text_proj.dim() > 2 else text_proj
            
            vision_norm = vision_proj / vision_proj.norm()
            text_norm = text_proj / text_proj.norm()
            
            sim = torch.cosine_similarity(vision_norm, text_norm).item()
            
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
    
    print("   🎤 Listening... (3 seconds)")
    frames = []
    for _ in range(0, int(16000 / 1024 * 3)):
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
print("Show objects and I'll ask what they are!\n")
print("Press Ctrl+C to stop\n")

try:
    while True:
        print("\n" + "-"*70)
        
        # Capture scene
        vision_emb = capture_scene()
        
        # Try to identify
        match, confidence = find_match(vision_emb)
        
        if match and confidence > CONFIDENCE_THRESHOLD:
            print(f"✓ I see: {match} (confidence: {confidence:.2f})")
        
        else:
            # Ask what it is
            question = "What is that?"
            answer = ask_question(question)
            
            if answer and len(answer) > 2:
                # Store as text embedding
                text_emb = text_encoder.encode(answer, convert_to_tensor=True).to(device)
                known_objects[answer] = text_emb
                print(f"   ✓ Learned: '{answer}'")
            else:
                print("   (Didn't catch that)")
        
        # Wait before next check
        print("\nWaiting 10 seconds...")
        time.sleep(10)

except KeyboardInterrupt:
    print("\n\nStopping Arnold...")
    print(f"Learned {len(known_objects)} objects: {list(known_objects.keys())}")