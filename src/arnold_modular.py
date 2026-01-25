# arnold_modular.py - Continuous Learning Architecture
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
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
import re
import os
from datetime import datetime

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ObjectKnowledge:
    """Everything we know about a learned object"""
    name: str
    text_embedding: torch.Tensor
    vision_embeddings: List[torch.Tensor] = field(default_factory=list)
    learned_timestamp: float = 0.0
    last_seen: float = 0.0
    times_recognized: int = 0

@dataclass 
class LearningMoment:
    """Data captured during a learning session - for later LoRA training"""
    object_name: str
    frames: List[np.ndarray]
    embeddings: List[torch.Tensor]
    timestamp: float

# ============================================================================
# VISION ENCODER
# ============================================================================

class VisionEncoder:
    def __init__(self, model_path: str, adapter_path: str, device: str = "cuda"):
        self.device = device
        
        print("  Loading base V-JEPA 2...")
        base_model = AutoModel.from_pretrained(model_path)
        
        print("  Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
    
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Encode video frames to embedding"""
        inputs = self.processor(videos=list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state.mean(dim=1)
        
        return embedding

# ============================================================================
# CONTINUOUS CAMERA
# ============================================================================

class ContinuousCamera:
    """Streams frames continuously for fluid capture"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.frame_buffer = []
        self.buffer_size = 64  # Frames needed for V-JEPA
    
    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.frame_buffer = []
    
    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get single frame"""
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def get_clip(self, num_frames: int = 64) -> Optional[np.ndarray]:
        """Capture a clip of frames"""
        frames = []
        for _ in range(num_frames):
            frame = self.get_frame()
            if frame is not None:
                frames.append(frame)
        
        if len(frames) == num_frames:
            return np.array(frames)
        return None
    
    def update_buffer(self):
        """Add frame to rolling buffer"""
        frame = self.get_frame()
        if frame is not None:
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
    
    def get_buffer_clip(self) -> Optional[np.ndarray]:
        """Get current buffer as clip (if full)"""
        if len(self.frame_buffer) >= self.buffer_size:
            return np.array(self.frame_buffer[-self.buffer_size:])
        return None

# ============================================================================
# CHANGE DETECTOR
# ============================================================================

class ChangeDetector:
    """Detects scene changes and regions of interest"""
    
    def __init__(self, threshold: float = 0.93):
        self.threshold = threshold
        self.previous_embedding: Optional[torch.Tensor] = None
        self.previous_frame: Optional[np.ndarray] = None
        self.grid_size = 4
    
    def check_scene_change(self, current_embedding: torch.Tensor) -> tuple:
        """Check if scene changed significantly"""
        if self.previous_embedding is None:
            self.previous_embedding = current_embedding.clone()
            return True, 0.0
        
        sim = torch.cosine_similarity(
            current_embedding.flatten().unsqueeze(0),
            self.previous_embedding.flatten().unsqueeze(0)
        ).item()
        
        changed = sim < self.threshold
        
        if changed:
            self.previous_embedding = current_embedding.clone()
        
        return changed, sim
    
    def find_changed_region(self, current_frame: np.ndarray) -> Optional[tuple]:
        """Find which region of frame changed most"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return None
        
        # Convert to grayscale
        current_gray = np.mean(current_frame, axis=2)
        previous_gray = np.mean(self.previous_frame, axis=2)
        
        # Difference
        diff = np.abs(current_gray.astype(float) - previous_gray.astype(float))
        
        # Find max change region
        h, w = diff.shape
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        max_change = 0
        best_region = None
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                
                change = np.mean(diff[y1:y2, x1:x2])
                if change > max_change:
                    max_change = change
                    best_region = (x1, y1, x2, y2)
        
        self.previous_frame = current_frame.copy()
        
        if max_change > 10:
            return best_region
        return None
    
    def crop_to_region(self, frames: np.ndarray, region: tuple, padding: float = 0.5) -> np.ndarray:
        """Crop frames to region with padding"""
        x1, y1, x2, y2 = region
        h, w = frames.shape[1:3]
        
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        cropped = frames[:, y1:y2, x1:x2, :]
        
        resized = np.array([cv2.resize(f, (w, h)) for f in cropped])
        return resized

# ============================================================================
# OBJECT MEMORY
# ============================================================================

class ObjectMemory:
    """Stores and recognizes learned objects"""
    
    def __init__(self, text_encoder, device: str = "cuda"):
        self.text_encoder = text_encoder
        self.device = device
        self.objects: Dict[str, ObjectKnowledge] = {}
    
    def add_views(self, name: str, embeddings: List[torch.Tensor]):
        """Add views of an object"""
        if name not in self.objects:
            text_emb = self.text_encoder.encode(name, convert_to_tensor=True).to(self.device)
            self.objects[name] = ObjectKnowledge(
                name=name,
                text_embedding=text_emb,
                learned_timestamp=time.time()
            )
        
        for emb in embeddings:
            self.objects[name].vision_embeddings.append(emb.clone())
        
        self.objects[name].last_seen = time.time()
    
    def recognize(self, embedding: torch.Tensor, min_views: int = 3) -> tuple:
        """
        Try to recognize object.
        Returns: (name, confidence, margin) or (None, 0, 0)
        """
        scores = {}
        
        for name, obj in self.objects.items():
            if len(obj.vision_embeddings) < min_views:
                continue
            
            # Compare to all stored views, take best matches
            view_scores = []
            for stored in obj.vision_embeddings:
                sim = torch.cosine_similarity(
                    embedding.flatten().unsqueeze(0),
                    stored.flatten().unsqueeze(0)
                ).item()
                view_scores.append(sim)
            
            # Average of top 3 scores
            view_scores.sort(reverse=True)
            avg = sum(view_scores[:3]) / min(3, len(view_scores))
            scores[name] = avg
        
        if not scores:
            return None, 0.0, 0.0
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = sorted_scores[0]
        
        margin = 0.0
        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1][1]
        
        return best_name, best_score, margin
    
    def save(self, path: str = "data/object_memory.json"):
        data = {}
        for name, obj in self.objects.items():
            data[name] = {
                'num_views': len(obj.vision_embeddings),
                'learned': obj.learned_timestamp,
                'last_seen': obj.last_seen,
                'times_recognized': obj.times_recognized
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str = "data/object_memory.json") -> int:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for name, info in data.items():
                text_emb = self.text_encoder.encode(name, convert_to_tensor=True).to(self.device)
                self.objects[name] = ObjectKnowledge(
                    name=name,
                    text_embedding=text_emb,
                    learned_timestamp=info['learned'],
                    last_seen=info['last_seen'],
                    times_recognized=info.get('times_recognized', 0)
                )
            return len(self.objects)
        except FileNotFoundError:
            return 0

# ============================================================================
# TRAINING DATA COLLECTOR (for later LoRA training)
# ============================================================================

class TrainingDataCollector:
    """Collects data for overnight LoRA training"""
    
    def __init__(self, save_dir: str = "./data/training"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.session_data: List[LearningMoment] = []
    
    def record(self, object_name: str, frames: List[np.ndarray], embeddings: List[torch.Tensor]):
        """Record a learning moment"""
        moment = LearningMoment(
            object_name=object_name,
            frames=frames,
            embeddings=[e.cpu() for e in embeddings],
            timestamp=time.time()
        )
        self.session_data.append(moment)
        print(f"      📝 Recorded training data: {object_name} ({len(frames)} frames, {len(embeddings)} embeddings)")
    
    def save_session(self):
        """Save session data for later training"""
        if not self.session_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"session_{timestamp}.pt")
        
        torch.save(self.session_data, path)
        print(f"✓ Saved training data: {path} ({len(self.session_data)} learning moments)")

# ============================================================================
# VOICE INTERFACE
# ============================================================================

class VoiceInterface:
    def __init__(self):
        print("  Loading Whisper (small)...")
        self.whisper = whisper.load_model("small")
    
    def ask(self, question: str) -> str:
        """Ask question and get voice response with typed fallback"""
        print(f"\n🤖 ARNOLD: {question}")
        input("   Press ENTER when ready... ")
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                       input=True, frames_per_buffer=1024)
        
        print("   🎤 Speak now! (5 seconds)")
        frames = []
        for _ in range(0, int(16000 / 1024 * 5)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        print("   ✓ Got it")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
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
        
        result = self.whisper.transcribe(audio_data)
        heard = result['text'].strip()
        
        print(f"   I heard: '{heard}'")
        correction = input("   Press ENTER if correct, or type correction: ").strip()
        
        return correction if correction else heard

# ============================================================================
# CONTINUOUS LEARNING ENGINE
# ============================================================================

class ContinuousLearner:
    """Handles the continuous learning loop for a single object"""
    
    def __init__(self, encoder: VisionEncoder, camera: ContinuousCamera, 
                 change_detector: ChangeDetector, device: str = "cuda"):
        self.encoder = encoder
        self.camera = camera
        self.change_detector = change_detector
        self.device = device
        
        # Learning parameters
        self.high_confidence = 0.88
        self.medium_confidence = 0.80
        self.required_margin = 0.06
        self.min_views_for_confidence = 5
        self.capture_interval = 0.5  # Seconds between captures
    
    def learn_object(self, object_name: str, memory: ObjectMemory) -> tuple:
        """
        Continuously capture views until confident.
        Returns: (embeddings_collected, frames_collected)
        """
        print(f"\n   📚 Learning '{object_name}'...")
        print(f"      Show me the object - rotate it, move it around")
        print(f"      I'll tell you when I've got it\n")
        
        collected_embeddings = []
        collected_frames = []
        last_capture = 0
        view_count = 0
        
        while True:
            # Capture at intervals
            now = time.time()
            if now - last_capture < self.capture_interval:
                # Update buffer between captures
                self.camera.update_buffer()
                time.sleep(0.05)
                continue
            
            last_capture = now
            
            # Get clip from buffer
            clip = self.camera.get_buffer_clip()
            if clip is None:
                continue
            
            # Find changed region and crop
            region = self.change_detector.find_changed_region(clip[-1])
            if region:
                clip = self.change_detector.crop_to_region(clip, region)
            
            # Encode
            embedding = self.encoder.encode_frames(clip)
            collected_embeddings.append(embedding)
            collected_frames.append(clip[-1])  # Save last frame for training
            view_count += 1
            
            # Add to memory immediately
            memory.add_views(object_name, [embedding])
            
            # Check confidence
            if view_count >= self.min_views_for_confidence:
                name, conf, margin = memory.recognize(embedding, min_views=3)
                
                print(f"      View {view_count}: confidence {conf:.2f}, margin {margin:.3f}")
                
                if name == object_name and conf > self.high_confidence and margin > self.required_margin:
                    print(f"\n   ✓ Got it! I know what a '{object_name}' looks like now.")
                    print(f"      Collected {view_count} views, confidence: {conf:.2f}")
                    return collected_embeddings, collected_frames
                
                elif view_count >= 10 and conf > self.medium_confidence:
                    print(f"\n   ✓ Good enough! Collected {view_count} views.")
                    return collected_embeddings, collected_frames
                
                elif view_count >= 15:
                    print(f"\n   ⚠ Captured {view_count} views - moving on")
                    return collected_embeddings, collected_frames
            else:
                print(f"      View {view_count} captured...")

# ============================================================================
# MAIN ARNOLD SYSTEM
# ============================================================================

class WorkshopArnold:
    def __init__(self):
        print("="*70)
        print("WORKSHOP ARNOLD - Continuous Learning")
        print("="*70)
        print("\nInitializing...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")
        
        self.encoder = VisionEncoder(
            "./models/base/vjepa2",
            "./models/adapters/workshop_lora_20260117_041817",
            self.device
        )
        
        print("  Loading text encoder...")
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.camera = ContinuousCamera()
        self.change_detector = ChangeDetector(threshold=0.93)
        self.memory = ObjectMemory(self.text_encoder, self.device)
        self.voice = VoiceInterface()
        self.training_collector = TrainingDataCollector()
        
        self.learner = ContinuousLearner(
            self.encoder, self.camera, self.change_detector, self.device
        )
        
        # Load previous memory
        loaded = self.memory.load()
        if loaded:
            print(f"\n✓ Loaded {loaded} objects from previous session:")
            for name, obj in self.memory.objects.items():
                print(f"    - {name} ({len(obj.vision_embeddings)} views)")
        else:
            print("\n✓ Starting fresh session")
        
        print("\n" + "="*70)
    
    def run(self):
        print("\nContinuous Learning Mode")
        print("Show me objects and I'll learn what they are!")
        print("Press Ctrl+C to stop\n")
        
        self.camera.start()
        
        try:
            while True:
                print("-" * 50)
                
                # Fill buffer
                print("  👁 Watching...")
                for _ in range(64):
                    self.camera.update_buffer()
                
                clip = self.camera.get_buffer_clip()
                if clip is None:
                    continue
                
                # Check for change
                region = self.change_detector.find_changed_region(clip[-1])
                if region:
                    focused_clip = self.change_detector.crop_to_region(clip, region)
                else:
                    focused_clip = clip
                
                embedding = self.encoder.encode_frames(focused_clip)
                changed, sim = self.change_detector.check_scene_change(embedding)
                
                if not changed:
                    time.sleep(1)
                    continue
                
                print(f"  ✓ Change detected (similarity: {sim:.3f})")
                
                # Try to recognize
                name, conf, margin = self.memory.recognize(embedding)
                
                if name and conf > 0.85 and margin > 0.06:
                    print(f"\n✓ I recognize this: **{name}** (conf: {conf:.2f})")
                    self.memory.objects[name].times_recognized += 1
                    self.memory.objects[name].last_seen = time.time()
                
                elif name and conf > 0.75 and margin > 0.04:
                    print(f"\n⚠ I think this might be '{name}' ({conf:.0%} confident)")
                    confirm = input("   Is that correct? (y/n): ").strip().lower()
                    
                    if confirm == 'y':
                        print(f"   ✓ Great! Adding this view to '{name}'")
                        self.memory.add_views(name, [embedding])
                    else:
                        self._learn_new_object()
                
                else:
                    self._learn_new_object()
                
                time.sleep(2)
        
        except KeyboardInterrupt:
            self._shutdown()
    
    def _learn_new_object(self):
        """Learn a new object through continuous capture"""
        answer = self.voice.ask("What is that object?")
        cleaned = self._clean_name(answer)
        
        if not self._is_valid_name(cleaned):
            print(f"   ⚠ Invalid name, skipping")
            return
        
        # Continuous learning
        embeddings, frames = self.learner.learn_object(cleaned, self.memory)
        
        # Save for later LoRA training
        self.training_collector.record(cleaned, frames, embeddings)
    
    def _shutdown(self):
        print("\n\nSession ended")
        self.camera.stop()
        
        self.memory.save()
        print(f"✓ Saved {len(self.memory.objects)} objects to memory")
        
        self.training_collector.save_session()
        
        print(f"\nKnown objects:")
        for name, obj in self.memory.objects.items():
            print(f"  - {name} ({len(obj.vision_embeddings)} views, recognized {obj.times_recognized}x)")
    
    @staticmethod
    def _clean_name(text: str) -> str:
        text = text.lower()
        for phrase in ["that's a", "that is a", "it's a", "it is a",
                       "this is a", "this is an", "that's an", "it's an",
                       "the ", "a ", "an "]:
            text = text.replace(phrase, "")
        text = re.sub(r'[.,!?;:]', '', text)
        return ' '.join(text.split()).strip()
    
    @staticmethod
    def _is_valid_name(name: str) -> bool:
        if len(name) < 2:
            return False
        if name in ['um', 'uh', 'the', 'what', 'where', 'on a', 'in a', 'a']:
            return False
        return any(c.isalpha() for c in name)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    arnold = WorkshopArnold()
    arnold.run()