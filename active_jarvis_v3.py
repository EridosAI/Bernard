# active_jarvis_v3.py - Better contextual questions
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
print("ACTIVE JARVIS - v3 (Contextual Questions)")
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
previous_scene = None
CONFIDENCE_THRESHOLD = 0.15
CHANGE_THRESHOLD = 0.93

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

def detect_properties(vision_emb):
    """Detect visible properties for better questions"""
    properties = []
    
    # Test colors
    colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple']
    best_color = None
    best_score = 0
    
    for color in colors:
        text_emb = text_encoder.encode(color, convert_to_tensor=True).to(device)
        
        with torch.no_grad():
            vision_proj, text_proj = alignment_head(vision_emb, text_emb.unsqueeze(0))
            vision_proj = vision_proj.squeeze(1) if vision_proj.dim() > 2 else vision_proj
            text_proj = text_proj.squeeze(1) if text_proj.dim() > 2 else text_proj
            
            vision_norm = vision_proj / vision_proj.norm()
            text_norm = text_proj / text_proj.norm()
            score = torch.cosine_similarity(vision_norm, text_norm).item()
        
        if score > best_score:
            best_score = score
            best_color = color
    
    if best_score > 0.05:  # Lower threshold
        properties.append(best_color)
    
    # Test shapes
    shapes = ['cylindrical', 'round', 'rectangular', 'square']
    best_shape = None
    best_score = 0
    
    for shape in shapes:
        text_emb = text_encoder.encode(shape, convert_to_tensor=True).to(device)
        
        with torch.no_grad():
            vision_proj, text_proj = alignment_head(vision_emb, text_emb.unsqueeze(0))
            vision_proj = vision_proj.squeeze(1) if vision_proj.dim() > 2 else vision_proj
            text_proj = text_proj.squeeze(1) if text_proj.dim() > 2 else text_proj
            
            vision_norm = vision_proj / vision_proj.norm()
            text_norm = text_proj / text_proj.norm()
            score = torch.cosine_similarity(vision_norm, text_norm).item()
        
        if score > best_score:
            best_score = score
            best_shape = shape
    
    if best_score > 0.05:  # Lower threshold
        properties.append(best_shape)
    
    return properties

def scene_changed(current_emb, previous_emb):
    """Check if scene changed significantly"""
    if previous_emb is None:
        return True
    
    sim = torch.cosine_similarity(current_emb, previous_emb).item()
    print(f"  Scene similarity: {sim:.3f} (threshold: {CHANGE_THRESHOLD})")
    
    return sim < CHANGE_THRESHOLD

def find_match(vision_emb):
    """Search known objects using alignment head"""
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

def ask_question(properties):
    """Ask contextual question with properties"""
    if properties:
        description = " ".join(properties)
        question = f"What is that {description} object?"
    else:
        question = "What new object did you just show me?"
    
    print(f"\n🤖 JARVIS: {question}")
    
    # Wait for user to be ready
    input("   Press ENTER when ready to answer... ")
    
    # Record answer
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    print("   🎤 Recording NOW - speak!")
    frames = []
    for _ in range(0, int(16000 / 1024 * 3)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    
    print("   ✓ Got it")
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
    
    print(f"   👤 You said: '{answer}'")
    return answer

def clean_response(answer):
    """Extract just the object name"""
    answer = answer.lower()
    answer = answer.replace("that's a", "").replace("that is a", "")
    answer = answer.replace("it's a", "").replace("it is a", "")
    answer = answer.replace("this is a", "").replace("this is", "")
    answer = answer.replace("that's", "").replace("it's", "")
    answer = answer.strip()
    return answer

# Main loop
print("\nStarting active monitoring...")
print("I'll only ask when I notice something NEW in the scene!")
print("\nPress Ctrl+C to stop\n")

while True:
    try:
        print("\n" + "-"*70)
        
        # Capture scene
        vision_emb = capture_scene()
        
        # Check if scene changed
        if not scene_changed(vision_emb, previous_scene):
            print("  (No change detected - scene looks the same)")
            previous_scene = vision_emb
            time.sleep(10)
            continue
        
        print("  ✓ Scene changed - analyzing...")
        
        # Try to identify
        match, confidence = find_match(vision_emb)
        
        if match and confidence > CONFIDENCE_THRESHOLD:
            print(f"✓ I see: {match} (confidence: {confidence:.2f})")
        
        else:
            # Detect properties for better question
            properties = detect_properties(vision_emb)
            
            if properties:
                print(f"  Properties detected: {', '.join(properties)}")
            
            # Ask what it is
            answer = ask_question(properties)
            
            if answer and len(answer) > 2:
                cleaned = clean_response(answer)
                
                # Validate - skip if too short or nonsense
                if len(cleaned) < 3:
                    print(f"   ⚠ '{cleaned}' seems too short - skipping")
                elif cleaned.lower() in ['on a', 'the', 'a', 'um', 'uh']:
                    print(f"   ⚠ Ignoring incomplete phrase")
                else:
                    # Store as text embedding
                    text_emb = text_encoder.encode(cleaned, convert_to_tensor=True).to(device)
                    known_objects[cleaned] = text_emb
                    print(f"   ✓ Learned: '{cleaned}'")
            else:
                print("   ⚠ Didn't catch that - try again")
        
        # Update previous scene
        previous_scene = vision_emb
        
        # Wait before next check
        print("\nWaiting 10 seconds...")
        time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\nStopping JARVIS...")
        print(f"\nLearned {len(known_objects)} objects:")
        for obj in known_objects.keys():
            print(f"  - {obj}")
        break
    
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        continue