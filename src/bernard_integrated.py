# bernard_integrated.py - Florence-2 + V-JEPA Integrated System
"""
Two-layer architecture:
  Layer 1: Florence-2 - General world knowledge (what objects are in scene)
  Layer 2: V-JEPA - Specific workshop knowledge (which specific object is this)
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
import re
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Whisper for voice
import whisper
import pyaudio
import wave
import soundfile as sf

# Models
from transformers import AutoModel, AutoVideoProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """A single object detected by Florence"""
    label: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
@dataclass
class ObjectMemory:
    """Stored knowledge about a specific object"""
    name: str                    # User's name for it ("Sony remote")
    category: str                # Florence's category ("remote control")
    embeddings: List[torch.Tensor] = field(default_factory=list)
    learned_at: float = 0.0
    last_seen: float = 0.0
    times_seen: int = 0

@dataclass
class RecognitionResult:
    """Result of trying to recognize an object"""
    florence_label: str          # What Florence calls it
    specific_name: Optional[str] # What we've learned to call it
    confidence: float
    is_new: bool
    bbox: Tuple[int, int, int, int]

# ============================================================================
# FLORENCE-2 (General Knowledge Layer)
# ============================================================================

class FlorenceDetector:
    """Handles object detection using Florence-2"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("  Loading Florence-2...")
        
        model_id = "microsoft/Florence-2-large"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
        print("  ✓ Florence-2 loaded")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect all objects in frame"""
        # Convert to PIL
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(frame)
        
        # Run detection
        prompt = "<OD>"
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device, torch.float16) if v.dtype == torch.float32 else v.to(self.device) 
                  for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            use_cache=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        result = self.processor.post_process_generation(
            generated_text,
            task="<OD>",
            image_size=(pil_image.width, pil_image.height)
        )
        
        # Parse detections
        detections = []
        if '<OD>' in result:
            labels = result['<OD>'].get('labels', [])
            bboxes = result['<OD>'].get('bboxes', [])
            
            for label, bbox in zip(labels, bboxes):
                detections.append(Detection(
                    label=label.lower(),
                    bbox=tuple(int(c) for c in bbox)
                ))
        
        return detections
    
    def caption(self, frame: np.ndarray) -> str:
        """Get a caption for the frame"""
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(frame)
        
        prompt = "<CAPTION>"
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device, torch.float16) if v.dtype == torch.float32 else v.to(self.device) 
                  for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            use_cache=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        result = self.processor.post_process_generation(
            generated_text,
            task="<CAPTION>",
            image_size=(pil_image.width, pil_image.height)
        )
        
        return result.get('<CAPTION>', '')

# ============================================================================
# V-JEPA (Specific Knowledge Layer)
# ============================================================================

class VJEPAEncoder:
    """Handles embeddings using V-JEPA"""
    
    def __init__(self, model_path: str, adapter_path: str, device: str = "cuda"):
        self.device = device
        print("  Loading V-JEPA 2...")
        
        base_model = AutoModel.from_pretrained(model_path)
        self.model = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
        
        print("  ✓ V-JEPA loaded")
    
    def encode_region(self, frames: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """Encode a specific region across frames"""
        x1, y1, x2, y2 = bbox
        h, w = frames.shape[1:3]
        
        # Add padding around bbox
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        # Crop and resize
        cropped = frames[:, y1:y2, x1:x2, :]
        resized = np.array([cv2.resize(f, (256, 256)) for f in cropped])
        
        # Encode
        inputs = self.processor(videos=list(resized), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state.mean(dim=1)
        
        return embedding.squeeze(0)
    
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Encode full frames"""
        inputs = self.processor(videos=list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state.mean(dim=1)
        
        return embedding.squeeze(0)

# ============================================================================
# OBJECT MEMORY STORE
# ============================================================================

class MemoryStore:
    """Stores and retrieves learned object knowledge"""
    
    def __init__(self, save_path: str = "data/object_memory_integrated.json"):
        self.save_path = save_path
        self.embeddings_path = save_path.replace('.json', '_embeddings.pt')
        self.objects: Dict[str, ObjectMemory] = {}
    
    def add_object(self, name: str, category: str, embedding: torch.Tensor):
        """Add or update an object"""
        if name not in self.objects:
            self.objects[name] = ObjectMemory(
                name=name,
                category=category,
                learned_at=time.time()
            )
        
        self.objects[name].embeddings.append(embedding.cpu().clone())
        self.objects[name].last_seen = time.time()
        self.objects[name].times_seen += 1
    
    def find_match(self, category: str, embedding: torch.Tensor, 
                   threshold: float = 0.82, margin: float = 0.05) -> Tuple[Optional[str], float]:
        """
        Find a matching object of the same category.
        Returns (name, confidence) or (None, 0)
        """
        candidates = {name: obj for name, obj in self.objects.items() 
                      if obj.category == category and len(obj.embeddings) >= 1}
        
        if not candidates:
            return None, 0.0
        
        scores = {}
        for name, obj in candidates.items():
            # Compare to stored embeddings
            max_sim = 0.0
            for stored_emb in obj.embeddings:
                sim = torch.cosine_similarity(
                    embedding.cpu().flatten().unsqueeze(0),
                    stored_emb.flatten().unsqueeze(0)
                ).item()
                max_sim = max(max_sim, sim)
            scores[name] = max_sim
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = sorted_scores[0]
        
        # Check margin if multiple candidates
        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            actual_margin = best_score - second_score
            if best_score > threshold and actual_margin > margin:
                return best_name, best_score
        elif best_score > threshold:
            return best_name, best_score
        
        return None, best_score
    
    def get_all_of_category(self, category: str) -> List[str]:
        """Get all learned objects of a category"""
        return [name for name, obj in self.objects.items() if obj.category == category]
    
    def save(self):
        """Save memory to disk"""
        # Save metadata
        data = {}
        for name, obj in self.objects.items():
            data[name] = {
                'category': obj.category,
                'num_embeddings': len(obj.embeddings),
                'learned_at': obj.learned_at,
                'last_seen': obj.last_seen,
                'times_seen': obj.times_seen
            }
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings
        embeddings_data = {name: obj.embeddings for name, obj in self.objects.items()}
        torch.save(embeddings_data, self.embeddings_path)
        
        print(f"✓ Saved {len(self.objects)} objects to memory")
    
    def load(self) -> int:
        """Load memory from disk"""
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            embeddings_data = torch.load(self.embeddings_path, weights_only=False)
            
            for name, info in data.items():
                self.objects[name] = ObjectMemory(
                    name=name,
                    category=info['category'],
                    embeddings=embeddings_data.get(name, []),
                    learned_at=info['learned_at'],
                    last_seen=info['last_seen'],
                    times_seen=info['times_seen']
                )
            
            return len(self.objects)
        except FileNotFoundError:
            return 0

# ============================================================================
# VOICE INTERFACE
# ============================================================================

class VoiceInterface:
    """Handles speech recognition"""
    
    def __init__(self):
        print("  Loading Whisper...")
        self.whisper = whisper.load_model("small")
        print("  ✓ Whisper loaded")
    
    def ask(self, question: str) -> str:
        """Ask a question and get voice response"""
        print(f"\n🤖 BERNARD: {question}")
        input("   Press ENTER when ready to speak...")
        
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
        
        result = self.whisper.transcribe(audio_data)
        heard = result['text'].strip()
        
        print(f"   I heard: '{heard}'")
        correction = input("   Press ENTER if correct, or type correction: ").strip()
        
        return self._clean_name(correction if correction else heard)
    
    @staticmethod
    def _clean_name(text: str) -> str:
        """Clean up object name"""
        text = text.lower()
        for phrase in ["that's a", "that is a", "it's a", "it is a",
                       "this is a", "this is an", "that's an", "it's an",
                       "that's my", "it's my", "this is my",
                       "the ", "a ", "an ", "my "]:
            if text.startswith(phrase):
                text = text[len(phrase):]
        text = re.sub(r'[.,!?;:]', '', text)
        return ' '.join(text.split()).strip()

# ============================================================================
# FOCUS SELECTOR
# ============================================================================

class FocusSelector:
    """Decides which detected object to focus on"""
    
    def __init__(self):
        self.previous_detections: List[Detection] = []
        self.ignored_categories = {'person', 'wall', 'floor', 'ceiling', 'desk', 'chair'}
    
    def select_focus(self, detections: List[Detection], 
                     frame_shape: Tuple[int, int]) -> Optional[Detection]:
        """
        Select which object to focus on.
        Priority:
        1. Objects near/overlapping hands
        2. New objects (not in previous frame)
        3. Objects closest to center
        """
        h, w = frame_shape[:2]
        center = (w // 2, h // 2)
        
        # Filter out ignored categories
        candidates = [d for d in detections if d.label not in self.ignored_categories]
        
        if not candidates:
            self.previous_detections = detections
            return None
        
        # Check for hand proximity
        hands = [d for d in detections if d.label == 'hand']
        if hands:
            for hand in hands:
                for obj in candidates:
                    if self._boxes_overlap(hand.bbox, obj.bbox):
                        self.previous_detections = detections
                        return obj
        
        # Check for new objects
        prev_labels = {d.label for d in self.previous_detections}
        new_objects = [d for d in candidates if d.label not in prev_labels]
        
        if new_objects:
            self.previous_detections = detections
            return new_objects[0]
        
        # Closest to center
        def dist_to_center(det):
            bx = (det.bbox[0] + det.bbox[2]) // 2
            by = (det.bbox[1] + det.bbox[3]) // 2
            return (bx - center[0])**2 + (by - center[1])**2
        
        candidates.sort(key=dist_to_center)
        self.previous_detections = detections
        return candidates[0] if candidates else None
    
    @staticmethod
    def _boxes_overlap(box1, box2, margin=30):
        """Check if two boxes overlap (with margin)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Expand box1 by margin
        x1_1 -= margin
        y1_1 -= margin
        x2_1 += margin
        y2_1 += margin
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

# ============================================================================
# MAIN INTEGRATED SYSTEM
# ============================================================================

class WorkshopBernard:
    """Integrated Florence + V-JEPA Workshop Assistant"""

    def __init__(self):
        print("=" * 70)
        print("WORKSHOP BERNARD - Integrated Vision System")
        print("=" * 70)
        print("\nInitializing...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")
        
        # Load models
        self.florence = FlorenceDetector(self.device)
        
        self.vjepa = VJEPAEncoder(
            "./models/base/vjepa2",
            "./models/adapters/workshop_lora_20260117_041817",
            self.device
        )
        
        self.voice = VoiceInterface()
        
        # Memory and focus
        self.memory = MemoryStore()
        self.focus = FocusSelector()
        
        # Camera
        self.cap = None
        self.frame_buffer = []
        
        # Load previous memory
        loaded = self.memory.load()
        if loaded:
            print(f"\n✓ Loaded {loaded} objects from memory:")
            for name, obj in self.memory.objects.items():
                print(f"    - {name} ({obj.category}, {len(obj.embeddings)} views)")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """Main loop"""
        print("\nWorkshop Vision System Active")
        print("Show me objects and I'll learn to recognize them!")
        print("Press Ctrl+C to stop\n")
        
        self.cap = cv2.VideoCapture(0)
        
        try:
            while True:
                print("-" * 50)
                
                # Capture frames for V-JEPA (need 64)
                print("  👁 Watching...")
                frames = self._capture_frames(64)
                
                if frames is None:
                    continue
                
                current_frame = frames[-1]
                
                # Florence detection
                print("  🔍 Detecting objects...")
                start = time.time()
                detections = self.florence.detect(current_frame)
                detect_time = (time.time() - start) * 1000
                
                if detections:
                    labels = [d.label for d in detections]
                    print(f"     Found ({detect_time:.0f}ms): {', '.join(labels)}")
                else:
                    print(f"     No objects detected ({detect_time:.0f}ms)")
                    time.sleep(1)
                    continue
                
                # Select focus object
                focus_obj = self.focus.select_focus(detections, current_frame.shape)
                
                if not focus_obj:
                    print("     Nothing to focus on")
                    time.sleep(1)
                    continue
                
                print(f"\n  🎯 Focusing on: {focus_obj.label}")
                
                # Get V-JEPA embedding for this region
                embedding = self.vjepa.encode_region(frames, focus_obj.bbox)
                
                # Try to match against known objects of this category
                match_name, confidence = self.memory.find_match(
                    focus_obj.label, embedding
                )
                
                if match_name:
                    # We recognize this specific object!
                    print(f"\n  ✓ I recognize this: **{match_name}** (conf: {confidence:.2f})")
                    self.memory.objects[match_name].last_seen = time.time()
                    self.memory.objects[match_name].times_seen += 1
                    
                    # Optionally add this view
                    if confidence < 0.90:
                        self.memory.add_object(match_name, focus_obj.label, embedding)
                        print(f"     (Added new view to strengthen memory)")
                
                else:
                    # Check if we have ANY objects of this category
                    known_of_type = self.memory.get_all_of_category(focus_obj.label)
                    
                    if known_of_type:
                        print(f"\n  🤔 I see a {focus_obj.label}, but it doesn't match:")
                        for name in known_of_type:
                            print(f"       - {name}")
                        
                        response = input("     Is this one of these? (name/n): ").strip().lower()
                        
                        if response == 'n' or response == 'no':
                            self._learn_new_object(focus_obj, embedding, frames)
                        elif response in [n.lower() for n in known_of_type]:
                            # Find the actual name
                            for name in known_of_type:
                                if name.lower() == response:
                                    self.memory.add_object(name, focus_obj.label, embedding)
                                    print(f"     ✓ Added new view to '{name}'")
                                    break
                        else:
                            self._learn_new_object(focus_obj, embedding, frames)
                    else:
                        # First object of this category
                        print(f"\n  🆕 First {focus_obj.label} I've seen!")
                        self._learn_new_object(focus_obj, embedding, frames)
                
                time.sleep(2)
        
        except KeyboardInterrupt:
            self._shutdown()
    
    def _capture_frames(self, num_frames: int = 64) -> Optional[np.ndarray]:
        """Capture frames from camera"""
        frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if len(frames) == num_frames:
            return np.array(frames)
        return None
    
    def _learn_new_object(self, detection: Detection, embedding: torch.Tensor, 
                          frames: np.ndarray):
        """Learn a new specific object"""
        # Ask what to call it
        name = self.voice.ask(f"What do you call this {detection.label}?")
        
        if len(name) < 2:
            print("     ⚠ Invalid name, skipping")
            return
        
        # Store it
        self.memory.add_object(name, detection.label, embedding)
        print(f"\n  ✓ Learned: **{name}** (category: {detection.label})")
        
        # Continuous learning - capture more views
        print(f"\n     Show me '{name}' from different angles...")
        print("     Press 'q' when done, or I'll stop after 10 views\n")
        
        views_captured = 1
        max_views = 10
        
        while views_captured < max_views:
            # Quick capture
            frames = self._capture_frames(64)
            if frames is None:
                continue
            
            # Check if object still there
            detections = self.florence.detect(frames[-1])
            matching = [d for d in detections if d.label == detection.label]
            
            if not matching:
                print(f"     Lost sight of {detection.label}...")
                break
            
            # Get embedding for closest match to original position
            best_det = min(matching, key=lambda d: 
                          abs(d.bbox[0] - detection.bbox[0]) + abs(d.bbox[1] - detection.bbox[1]))
            
            new_embedding = self.vjepa.encode_region(frames, best_det.bbox)
            self.memory.add_object(name, detection.label, new_embedding)
            views_captured += 1
            
            print(f"     View {views_captured} captured")
            
            # Check for quit
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        
        print(f"     ✓ Learned '{name}' with {views_captured} views")
    
    def _shutdown(self):
        """Clean shutdown"""
        print("\n\nShutting down...")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.memory.save()
        
        print(f"\nKnown objects:")
        for name, obj in self.memory.objects.items():
            print(f"  - {name} ({obj.category}, {len(obj.embeddings)} views, seen {obj.times_seen}x)")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    bernard = WorkshopBernard()
    bernard.run()