# jarvis_integrated.py - Florence-2 + V-JEPA Integrated System
"""
Two-layer architecture:
  Layer 1: Florence-2 - General world knowledge (what objects are in scene)
  Layer 2: V-JEPA - Specific workshop knowledge (which specific object is this)
  
v2: Added spatial persistence - tracks objects through rotation/occlusion
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
from typing import Optional, Dict, List, Tuple, Any
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

# Episode Memory
from episode_memory import EpisodeMemory

# Novelty Scoring
from novelty_scorer import NoveltyScorer

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingFrame:
    """Single frame with metadata for training"""
    frame: np.ndarray
    bbox: Tuple[int, int, int, int]
    object_name: str
    category: str
    timestamp: float

@dataclass
class TrainingSession:
    """Collection of training data from one session"""
    frames: List[TrainingFrame] = field(default_factory=list)
    session_start: float = 0.0
    session_end: float = 0.0

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
# SPATIAL TRACKER
# ============================================================================

class SpatialTracker:
    """Tracks objects through space even when labels change"""
    
    def __init__(self):
        self.tracked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.tracked_category: Optional[str] = None
        self.frames_since_seen: int = 0
        self.max_frames_lost: int = 5  # How many frames to keep tracking without detection
    
    def start_tracking(self, bbox: Tuple[int, int, int, int], category: str):
        """Start tracking an object"""
        self.tracked_bbox = bbox
        self.tracked_category = category
        self.frames_since_seen = 0
    
    def stop_tracking(self):
        """Stop tracking"""
        self.tracked_bbox = None
        self.tracked_category = None
        self.frames_since_seen = 0
    
    def update(self, detections: List[Detection]) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracking based on new detections.
        Returns the best bbox to use for the tracked object.
        """
        if self.tracked_bbox is None:
            return None
        
        # First, try to find the same category at similar position
        same_category = [d for d in detections if d.label == self.tracked_category]
        
        if same_category:
            # Find the one closest to tracked position
            best = self._find_closest(same_category, self.tracked_bbox)
            if best and self._boxes_overlap(best.bbox, self.tracked_bbox, margin=100):
                self.tracked_bbox = best.bbox
                self.frames_since_seen = 0
                return best.bbox
        
        # Category not found - look for ANYTHING at that position
        # This handles rotation, occlusion, etc.
        nearby = [d for d in detections if self._boxes_overlap(d.bbox, self.tracked_bbox, margin=80)]
        
        if nearby:
            # Something is there! Use its bbox
            best = self._find_closest(nearby, self.tracked_bbox)
            if best:
                # Update tracked position to follow the object
                self.tracked_bbox = best.bbox
                self.frames_since_seen = 0
                return best.bbox
        
        # Nothing found at position
        self.frames_since_seen += 1
        
        if self.frames_since_seen > self.max_frames_lost:
            # Lost it for too long
            return None
        
        # Return last known position (object might be briefly undetected)
        return self.tracked_bbox
    
    def _find_closest(self, detections: List[Detection], 
                      target_bbox: Tuple[int, int, int, int]) -> Optional[Detection]:
        """Find detection closest to target position"""
        if not detections:
            return None
        
        def center_dist(det):
            cx1 = (det.bbox[0] + det.bbox[2]) / 2
            cy1 = (det.bbox[1] + det.bbox[3]) / 2
            cx2 = (target_bbox[0] + target_bbox[2]) / 2
            cy2 = (target_bbox[1] + target_bbox[3]) / 2
            return (cx1 - cx2)**2 + (cy1 - cy2)**2
        
        return min(detections, key=center_dist)
    
    @staticmethod
    def _boxes_overlap(box1: Tuple, box2: Tuple, margin: int = 50) -> bool:
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
# OBJECT MEMORY STORE
# ============================================================================

class MemoryStore:
    """Stores and retrieves learned object knowledge"""
    
    def __init__(self, save_path: str = "object_memory_integrated.json"):
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
                   threshold: float = 0.82, margin: float = 0.05,
                   episode_memory: Optional[Any] = None) -> Tuple[Optional[str], float]:
        """
        Find a matching object of the same category.
        Returns (name, confidence) or (None, 0)
        
        Now with second-pass lookup: if no match found in the detected category,
        checks episode_memory for misclassification events where Florence's
        category matches the current detection, then tries matching against those objects.
        """
        # First pass: standard category-based matching
        candidates = {name: obj for name, obj in self.objects.items() 
                      if obj.category == category and len(obj.embeddings) >= 1}
        
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
        
        # Check if we have a good match from first pass
        if scores:
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
        
        # Second pass: check for misclassified objects
        # If we reach here, either no candidates or no good match from first pass
        if episode_memory is not None:
            # Find misclassification events where Florence detected the current category
            misclassified_candidates = {}
            
            for episode in episode_memory.episodes:
                if episode.event_type == "misclassification":
                    for obj_sighting in episode.objects:
                        # obj_sighting.category is what Florence detected (wrong)
                        # obj_sighting.name is the actual object
                        if obj_sighting.category == category and obj_sighting.name in self.objects:
                            # This object was misclassified as the current category before
                            # Add it as a candidate for matching
                            actual_obj = self.objects[obj_sighting.name]
                            if len(actual_obj.embeddings) >= 1:
                                misclassified_candidates[obj_sighting.name] = actual_obj
            
            # Try matching against misclassified candidates
            if misclassified_candidates:
                for name, obj in misclassified_candidates.items():
                    # Skip if already checked in first pass
                    if name in scores:
                        continue
                    
                    # Compare to stored embeddings
                    max_sim = 0.0
                    for stored_emb in obj.embeddings:
                        sim = torch.cosine_similarity(
                            embedding.cpu().flatten().unsqueeze(0),
                            stored_emb.flatten().unsqueeze(0)
                        ).item()
                        max_sim = max(max_sim, sim)
                    scores[name] = max_sim
        
        # Final decision: check all candidates (first pass + second pass)
        if not scores:
            return None, 0.0
        
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
# TRAINING DATA COLLECTOR
# ============================================================================

class TrainingDataCollector:
    """Collects data during sessions for overnight LoRA training"""
    
    def __init__(self, save_dir: str = "./data/training"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.current_session = TrainingSession(session_start=time.time())
        self.frames_collected = 0
        
        # Separate tracking for confirmed vs uncertain frames
        self.confirmed_frames: List[TrainingFrame] = []
        self.uncertain_frames: List[TrainingFrame] = []
        
    def record_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                     object_name: str, category: str, confirmed: bool = True):
        """Record a single training frame
        
        Args:
            frame: The full frame
            bbox: Bounding box of the object
            object_name: User's name for the object
            category: Florence's category
            confirmed: True if Florence confirmed the category, False if uncertain
        """
        # Crop the object region
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        cropped = frame[y1:y2, x1:x2].copy()
        
        training_frame = TrainingFrame(
            frame=cropped,
            bbox=(x1, y1, x2, y2),
            object_name=object_name,
            category=category,
            timestamp=time.time()
        )
        
        if confirmed:
            self.confirmed_frames.append(training_frame)
        else:
            self.uncertain_frames.append(training_frame)
        
        self.frames_collected += 1
    
    def save_session(self):
        """Save current session to disk"""
        total_frames = len(self.confirmed_frames) + len(self.uncertain_frames)
        
        if total_frames == 0:
            print("  No training data to save")
            return None
        
        # Create session filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(self.save_dir, f"session_{timestamp}")
        os.makedirs(session_path, exist_ok=True)
        
        manifest = {
            'session_start': self.current_session.session_start,
            'session_end': time.time(),
            'total_confirmed': len(self.confirmed_frames),
            'total_uncertain': len(self.uncertain_frames),
            'objects': {}
        }
        
        # Save confirmed frames
        self._save_frames(self.confirmed_frames, session_path, "confirmed", manifest)
        
        # Save uncertain frames
        self._save_frames(self.uncertain_frames, session_path, "uncertain", manifest)
        
        # Save manifest
        manifest_path = os.path.join(session_path, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n📝 Training data saved: {session_path}")
        print(f"   Confirmed: {len(self.confirmed_frames)} frames")
        print(f"   Uncertain: {len(self.uncertain_frames)} frames (will review during dream)")
        
        # Detailed breakdown
        confirmed_objects = {}
        for tf in self.confirmed_frames:
            confirmed_objects[tf.object_name] = confirmed_objects.get(tf.object_name, 0) + 1
        
        uncertain_objects = {}
        for tf in self.uncertain_frames:
            uncertain_objects[tf.object_name] = uncertain_objects.get(tf.object_name, 0) + 1
        
        if confirmed_objects:
            print(f"   Confirmed breakdown:")
            for obj_name, count in confirmed_objects.items():
                print(f"     - {obj_name}: {count} frames")
        
        if uncertain_objects:
            print(f"   Uncertain breakdown:")
            for obj_name, count in uncertain_objects.items():
                print(f"     - {obj_name}: {count} frames (pending review)")
        
        return session_path
    
    def _save_frames(self, frames: List[TrainingFrame], session_path: str, 
                     frame_type: str, manifest: dict):
        """Save a list of frames to disk"""
        if not frames:
            return
        
        # Group by object
        objects_data = {}
        for tf in frames:
            if tf.object_name not in objects_data:
                objects_data[tf.object_name] = {
                    'category': tf.category,
                    'frames': []
                }
            objects_data[tf.object_name]['frames'].append(tf.frame)
        
        # Save each object's frames
        for obj_name, data in objects_data.items():
            # Create object directory with frame type subdirectory
            obj_dir = os.path.join(session_path, obj_name.replace(' ', '_'), frame_type)
            os.makedirs(obj_dir, exist_ok=True)
            
            # Save frames as images
            for i, frame in enumerate(data['frames']):
                frame_path = os.path.join(obj_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Update manifest
            if obj_name not in manifest['objects']:
                manifest['objects'][obj_name] = {
                    'category': data['category'],
                    'confirmed': 0,
                    'uncertain': 0
                }
            
            manifest['objects'][obj_name][frame_type] = len(data['frames'])
    
    def get_stats(self) -> dict:
        """Get collection statistics"""
        confirmed_objects = {}
        for tf in self.confirmed_frames:
            confirmed_objects[tf.object_name] = confirmed_objects.get(tf.object_name, 0) + 1
        
        uncertain_objects = {}
        for tf in self.uncertain_frames:
            uncertain_objects[tf.object_name] = uncertain_objects.get(tf.object_name, 0) + 1
        
        return {
            'confirmed': len(self.confirmed_frames),
            'uncertain': len(self.uncertain_frames),
            'confirmed_by_object': confirmed_objects,
            'uncertain_by_object': uncertain_objects
        }

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
        print(f"\n🤖 JARVIS: {question}")
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
                     frame_shape: Tuple[int, int],
                     novelty_scorer: Optional['NoveltyScorer'] = None,
                     episode_memory: Optional[Any] = None,
                     memory_store: Optional[Any] = None,
                     vjepa_encoder: Optional[Any] = None,
                     frames: Optional[np.ndarray] = None) -> Optional[Detection]:
        """
        Select which object to focus on.
        Priority:
        1. Objects near/overlapping hands
        2. Highest novelty object (if novelty_scorer provided)
        3. New objects (not in previous frame)
        4. Objects closest to center
        """
        h, w = frame_shape[:2]
        center = (w // 2, h // 2)
        
        # Filter out ignored categories
        candidates = [d for d in detections if d.label not in self.ignored_categories]
        
        if not candidates:
            self.previous_detections = detections
            return None
        
        # Check for hand proximity (highest priority)
        hands = [d for d in detections if d.label == 'hand']
        if hands:
            for hand in hands:
                for obj in candidates:
                    if self._boxes_overlap(hand.bbox, obj.bbox):
                        self.previous_detections = detections
                        return obj
        
        # Use novelty scoring if available
        if novelty_scorer and episode_memory and memory_store and vjepa_encoder and frames is not None:
            novelty_scores = {}  # index -> score
            
            for i, det in enumerate(candidates):
                # Get embedding for this detection
                try:
                    embedding = vjepa_encoder.encode_region(frames, det.bbox)
                    
                    # Try to match against known objects
                    match_name, confidence = memory_store.find_match(
                        det.label, embedding, episode_memory=episode_memory
                    )
                    
                    if match_name:
                        # Known object - use combined novelty
                        # Calculate normalized position
                        cx = (det.bbox[0] + det.bbox[2]) / 2 / w
                        cy = (det.bbox[1] + det.bbox[3]) / 2 / h
                        position = (cx, cy)
                        
                        novelty_data = novelty_scorer.get_combined_novelty(
                            match_name, confidence, position, episode_memory
                        )
                        novelty_scores[i] = novelty_data['combined_novelty']
                    else:
                        # Unknown object - high object novelty (no confidence)
                        object_novelty = novelty_scorer.get_object_novelty(0.0)  # 0 confidence = 1.0 novelty
                        scene_novelty = novelty_scorer.get_scene_novelty()
                        # Combine with higher weight on object novelty for unknown objects
                        novelty_scores[i] = 0.6 * object_novelty + 0.4 * scene_novelty
                except Exception as e:
                    # If novelty scoring fails, use fallback
                    print(f"       Warning: Novelty scoring failed for {det.label}: {e}")
                    novelty_scores[i] = 0.5  # Moderate novelty as fallback
            
            # Select object with highest novelty by index
            if novelty_scores:
                best_idx = max(novelty_scores, key=novelty_scores.get)
                self.previous_detections = detections
                return candidates[best_idx]
        
        # Fallback: Check for new objects
        prev_labels = {d.label for d in self.previous_detections}
        new_objects = [d for d in candidates if d.label not in prev_labels]
        
        if new_objects:
            self.previous_detections = detections
            return new_objects[0]
        
        # Final fallback: Closest to center
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

class WorkshopJARVIS:
    """Integrated Florence + V-JEPA Workshop Assistant"""
    
    def __init__(self):
        print("=" * 70)
        print("WORKSHOP JARVIS - Integrated Vision System v2")
        print("  (with spatial persistence)")
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
        
        # Memory, focus, and tracking
        self.memory = MemoryStore()
        self.focus = FocusSelector()
        self.tracker = SpatialTracker()
        self.training_collector = TrainingDataCollector()
        self.episodes = EpisodeMemory()
        self.novelty = NoveltyScorer()
        
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
    
    def _draw_debug_frame(self, frame: np.ndarray, detections: List[Detection], 
                          focus_obj: Optional[Detection] = None,
                          recognition: Optional[Tuple[str, float]] = None) -> np.ndarray:
        """Draw debug visualization on frame"""
        # Convert RGB to BGR for OpenCV display
        display = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on status
            if focus_obj and det.bbox == focus_obj.bbox:
                # Focused object - green, thick
                color = (0, 255, 0)
                thickness = 3
            elif det.label in self.focus.ignored_categories:
                # Ignored category - gray, thin
                color = (128, 128, 128)
                thickness = 1
            else:
                # Other objects - blue
                color = (255, 200, 0)
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Label background
            label = det.label
            if focus_obj and det.bbox == focus_obj.bbox and recognition:
                name, conf = recognition
                if name:
                    label = f"{name} ({conf:.2f})"
                else:
                    label = f"{det.label} (NEW)"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add status text
        status = "WATCHING"
        if focus_obj:
            status = f"FOCUS: {focus_obj.label}"
        cv2.putText(display, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\nWorkshop Vision System Active")
        print("Show me objects and I'll learn to recognize them!")
        print("Press Ctrl+C to stop")
        print("Debug window will show detections\n")
        
        self.cap = cv2.VideoCapture(0)
        
        # Create debug window
        cv2.namedWindow("JARVIS Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JARVIS Debug", 960, 720)
        
        try:
            while True:
                print("-" * 50)
                
                # Capture frames for V-JEPA (need 64)
                print("  👁 Watching...")
                frames = self._capture_frames(64)
                
                if frames is None:
                    continue
                
                current_frame = frames[-1]
                
                # Get full-scene embedding and update novelty
                scene_emb = self.vjepa.encode_frames(frames)
                self.novelty.update_scene(scene_emb.cpu().numpy())
                
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
                    # Show frame even without detections
                    display = cv2.cvtColor(current_frame.copy(), cv2.COLOR_RGB2BGR)
                    cv2.putText(display, "No objects detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("JARVIS Debug", display)
                    cv2.waitKey(1)
                    time.sleep(1)
                    continue
                
                # Select focus object with novelty scoring
                focus_obj = self.focus.select_focus(
                    detections, 
                    current_frame.shape,
                    novelty_scorer=self.novelty,
                    episode_memory=self.episodes,
                    memory_store=self.memory,
                    vjepa_encoder=self.vjepa,
                    frames=frames
                )
                
                if not focus_obj:
                    print("     Nothing to focus on")
                    display = self._draw_debug_frame(current_frame, detections)
                    cv2.imshow("JARVIS Debug", display)
                    cv2.waitKey(1)
                    time.sleep(1)
                    continue
                
                # Calculate novelty score for the focused object
                embedding_for_novelty = self.vjepa.encode_region(frames, focus_obj.bbox)
                match_name_for_novelty, confidence_for_novelty = self.memory.find_match(
                    focus_obj.label, embedding_for_novelty, episode_memory=self.episodes
                )
                
                if match_name_for_novelty:
                    # Known object - get combined novelty
                    h, w = current_frame.shape[:2]
                    cx = (focus_obj.bbox[0] + focus_obj.bbox[2]) / 2 / w
                    cy = (focus_obj.bbox[1] + focus_obj.bbox[3]) / 2 / h
                    position = (cx, cy)
                    novelty_data = self.novelty.get_combined_novelty(
                        match_name_for_novelty, confidence_for_novelty, position, self.episodes
                    )
                    novelty_score = novelty_data['combined_novelty']
                    print(f"\n  🎯 Focusing on: {focus_obj.label} (novelty: {novelty_score:.2f})")
                else:
                    # Unknown object - high novelty
                    object_novelty = self.novelty.get_object_novelty(0.0)
                    scene_novelty = self.novelty.get_scene_novelty()
                    novelty_score = 0.6 * object_novelty + 0.4 * scene_novelty
                    print(f"\n  🎯 Focusing on: {focus_obj.label} (novelty: {novelty_score:.2f})")
                
                # Get V-JEPA embedding for this region
                embedding = self.vjepa.encode_region(frames, focus_obj.bbox)
                
                # Try to match against known objects of this category
                match_name, confidence = self.memory.find_match(
                    focus_obj.label, embedding, episode_memory=self.episodes
                )
                
                # Show debug frame with recognition result
                display = self._draw_debug_frame(
                    current_frame, detections, focus_obj, 
                    (match_name, confidence) if match_name else None
                )
                cv2.imshow("JARVIS Debug", display)
                cv2.waitKey(1)
                
                if match_name:
                    # We recognize this specific object!
                    print(f"\n  ✓ I recognize this: **{match_name}** (conf: {confidence:.2f})")
                    self.memory.objects[match_name].last_seen = time.time()
                    self.memory.objects[match_name].times_seen += 1

                    # Record episode
                    self.episodes.record_episode(
                        objects=[{
                            'name': match_name,
                            'category': focus_obj.label,
                            'confidence': confidence,
                            'bbox': focus_obj.bbox,
                            'is_focus': True
                        }],
                        event_type="observation"
                    )
                    print(f"     [Episode recorded]")  # ADD THIS LINE
                    
                    # Optionally add this view to strengthen memory
                    if confidence < 0.90:
                        self.memory.add_object(match_name, focus_obj.label, embedding)
                        print(f"     (Added new view to strengthen memory)")
                    
                    # Check for correction (non-blocking)
                    print(f"     [Press 'c' on debug window to correct if wrong]")
                    key = cv2.waitKey(1500) & 0xFF  # 1.5 second window to press 'c'
                    
                    if key == ord('c'):
                        self._correct_recognition(focus_obj, embedding, frames, 
                                                   wrong_name=match_name)
                
                else:
                    # Check if we have ANY objects of this category
                    known_of_type = self.memory.get_all_of_category(focus_obj.label)
                    
                    if known_of_type:
                        print(f"\n  🤔 I see a {focus_obj.label}, but it doesn't match:")
                        for i, name in enumerate(known_of_type, 1):
                            print(f"       {i}. {name}")
                        
                        if len(known_of_type) == 1:
                            response = input(f"     Is this '{known_of_type[0]}'? (y/n): ").strip().lower()
                            
                            if response in ['y', 'yes']:
                                self.memory.add_object(known_of_type[0], focus_obj.label, embedding)
                                print(f"     ✓ Added new view to '{known_of_type[0]}'")
                            else:
                                self._learn_new_object(focus_obj, embedding, frames)
                        else:
                            response = input("     Which one? (number/name/n for new): ").strip().lower()
                            
                            if response == 'n' or response == 'no':
                                self._learn_new_object(focus_obj, embedding, frames)
                            elif response.isdigit() and 1 <= int(response) <= len(known_of_type):
                                # User entered a number
                                idx = int(response) - 1
                                name = known_of_type[idx]
                                self.memory.add_object(name, focus_obj.label, embedding)
                                print(f"     ✓ Added new view to '{name}'")
                            elif response in [n.lower() for n in known_of_type]:
                                # User entered a name
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
        """Learn a new specific object with spatial persistence"""
        # Ask what to call it
        name = self.voice.ask(f"What do you call this {detection.label}?")
        
        if len(name) < 2:
            print("     ⚠ Invalid name, skipping")
            return
        
        # Check if this name already exists (regardless of category)
        if name in self.memory.objects:
            existing_obj = self.memory.objects[name]
            print(f"\n  ⚠ I already know an object called '{name}':")
            print(f"     Category: {existing_obj.category}")
            print(f"     Learned: {time.strftime('%Y-%m-%d %H:%M', time.localtime(existing_obj.learned_at))}")
            print(f"     Times seen: {existing_obj.times_seen}")
            
            response = input(f"     Is this the SAME '{name}'? (y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                # Same object! Add view to existing object
                print(f"\n  ✓ Adding view to existing '{name}'")
                
                # Check if categories match
                if existing_obj.category != detection.label:
                    print(f"     🔄 Interesting! Florence saw it as '{detection.label}' but I knew it as '{existing_obj.category}'")
                    print(f"     This will help me learn when Florence misclassifies!")
                    
                    # Record as misclassification episode for training
                    self.episodes.record_episode(
                        objects=[{
                            'name': name,
                            'category': detection.label,  # What Florence saw
                            'known_category': existing_obj.category,  # What we know it to be
                            'confidence': 1.0,
                            'bbox': detection.bbox,
                            'is_focus': True
                        }],
                        event_type="misclassification",
                        event_detail=f"Florence classified '{name}' as '{detection.label}' but it's actually a '{existing_obj.category}'"
                    )
                    print(f"     [Misclassification episode recorded for training]")
                
                # Add this view to the existing object
                self.memory.add_object(name, existing_obj.category, embedding)
                
                # Record training data with correct category
                self.training_collector.record_frame(
                    frames[-1], detection.bbox, name, existing_obj.category
                )
                
                # Continue with spatial learning (using existing category)
                print(f"\n     Show me '{name}' from different angles...")
                print("     (Click debug window and press 'q' to stop early)")
                print(f"     Auto-stops after 10 views or when object leaves frame\n")
                
                self._spatial_learning_loop(name, existing_obj.category, detection.bbox, embedding, frames)
                return
            else:
                # Different object with same name!
                print(f"\n  ℹ You have two objects with the same name '{name}'")
                new_name = input(f"     What should I call THIS {detection.label}? ").strip()
                
                if len(new_name) < 2:
                    print("     ⚠ Invalid name, skipping")
                    return
                
                name = new_name
                print(f"     Using new name: '{name}'")
        
        # Store initial view (either new name or user provided different name)
        self.memory.add_object(name, detection.label, embedding)
        print(f"\n  ✓ Learned: **{name}** (category: {detection.label})")
        
        # Record training data for initial frame
        self.training_collector.record_frame(
            frames[-1], detection.bbox, name, detection.label
        )
        
        # Start spatial tracking
        self.tracker.start_tracking(detection.bbox, detection.label)
        
        # Continuous learning with spatial persistence
        print(f"\n     Show me '{name}' from different angles...")
        print("     (Click debug window and press 'q' to stop early)")
        print(f"     Auto-stops after 10 views or when object leaves frame\n")
        
        views_captured = 1
        max_views = 10
        frames_without_category = 0  # Florence can't see the category
        max_frames_without_category = 3  # Allow brief gaps (rotation) but not removal
        last_good_embedding = embedding.clone()  # Track embedding continuity
        
        while views_captured < max_views:
            # Check for quit key - user must click debug window first
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 13 or key == 27:  # q, Enter, or Escape
                print("     Stopping early (key pressed)...")
                break
            
            # Quick capture
            frames = self._capture_frames(64)
            if frames is None:
                continue
            
            # Run Florence detection
            detections = self.florence.detect(frames[-1])
            
            # Update debug display during learning
            display = self._draw_debug_frame(frames[-1], detections, None, None)
            cv2.putText(display, f"LEARNING: {name} ({views_captured}/{max_views})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display, "Click here + press 'q' to stop", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("JARVIS Debug", display)
            
            # Check if Florence can still see this category
            category_detections = [d for d in detections if d.label == detection.label]
            florence_confirmed = len(category_detections) > 0
            
            if not florence_confirmed:
                frames_without_category += 1
                
                # Try spatial tracking - something might still be there
                tracked_bbox = self.tracker.update(detections)
                
                if tracked_bbox is not None:
                    # Something is there but Florence doesn't recognize the category
                    # Save as UNCERTAIN for dream consolidation
                    new_embedding = self.vjepa.encode_region(frames, tracked_bbox)
                    
                    sim_to_original = torch.cosine_similarity(
                        embedding.cpu().flatten().unsqueeze(0),
                        new_embedding.cpu().flatten().unsqueeze(0)
                    ).item()
                    
                    # Only save if somewhat similar (not completely wrong)
                    if sim_to_original > 0.25:
                        self.training_collector.record_frame(
                            frames[-1], tracked_bbox, name, detection.label, 
                            confirmed=False  # Uncertain!
                        )
                        print(f"     📦 Saved uncertain frame (sim: {sim_to_original:.2f}) - will review during dream")
                
                print(f"     ... can't see {detection.label} ({frames_without_category}/{max_frames_without_category})...")
                
                if frames_without_category >= max_frames_without_category:
                    print(f"     Object appears to be removed (Florence can't see {detection.label})")
                    break
                continue
            
            # Florence can see the category - reset counter
            frames_without_category = 0
            
            # Use spatial tracker to find the best bbox
            tracked_bbox = self.tracker.update(detections)
            
            if tracked_bbox is None:
                print(f"     Lost spatial tracking...")
                break
            
            # Get embedding for tracked region
            new_embedding = self.vjepa.encode_region(frames, tracked_bbox)
            
            # Check similarity to PREVIOUS good embedding (not just first)
            # This catches gradual drift to wrong object
            sim_to_last = torch.cosine_similarity(
                last_good_embedding.cpu().flatten().unsqueeze(0),
                new_embedding.cpu().flatten().unsqueeze(0)
            ).item()
            
            # Also check similarity to original
            sim_to_original = torch.cosine_similarity(
                embedding.cpu().flatten().unsqueeze(0),
                new_embedding.cpu().flatten().unsqueeze(0)
            ).item()
            
            # Accept if similar to either (allows gradual rotation)
            # But require at least SOME similarity to original (prevents drift to wrong object)
            if sim_to_last > 0.40 and sim_to_original > 0.30:
                self.memory.add_object(name, detection.label, new_embedding)
                views_captured += 1
                print(f"     View {views_captured} captured (sim: {sim_to_original:.2f}, drift: {sim_to_last:.2f})")
                
                # Record training data as CONFIRMED
                self.training_collector.record_frame(
                    frames[-1], tracked_bbox, name, detection.label,
                    confirmed=True
                )
                
                # Update last good embedding for drift detection
                last_good_embedding = new_embedding.clone()
            else:
                if sim_to_original < 0.30:
                    print(f"     Skipped (drifted too far from original: {sim_to_original:.2f})")
                else:
                    print(f"     Skipped (sudden change: {sim_to_last:.2f})")
        
        self.tracker.stop_tracking()
        print(f"     ✓ Learned '{name}' with {views_captured} views")
        
        # Record learning episode
        self.episodes.record_episode(
            objects=[{
                'name': name,
                'category': detection.label,
                'confidence': 1.0,
                'bbox': detection.bbox,
                'is_focus': True
            }],
            event_type="learning",
            event_detail=f"Learned new object '{name}' with {views_captured} views"
        )
    
    def _spatial_learning_loop(self, name: str, category: str, initial_bbox: Tuple[int, int, int, int],
                                initial_embedding: torch.Tensor, initial_frames: np.ndarray):
        """
        Spatial learning loop - captures multiple views of an object through rotation/movement.
        
        Args:
            name: Object name
            category: Florence category to track
            initial_bbox: Starting bounding box
            initial_embedding: Initial embedding for comparison
            initial_frames: Initial frame capture
        """
        views_captured = 1  # Already have initial view
        max_views = 10
        frames_without_category = 0
        max_frames_without_category = 3
        last_good_embedding = initial_embedding.clone()
        
        while views_captured < max_views:
            # Check for quit key
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 13 or key == 27:
                print("     Stopping early (key pressed)...")
                break
            
            # Capture frames
            frames = self._capture_frames(64)
            if frames is None:
                continue
            
            # Run detection
            detections = self.florence.detect(frames[-1])
            
            # Update debug display
            display = self._draw_debug_frame(frames[-1], detections, None, None)
            cv2.putText(display, f"LEARNING: {name} ({views_captured}/{max_views})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display, "Click here + press 'q' to stop", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("JARVIS Debug", display)
            
            # Check if Florence can still see the category
            category_detections = [d for d in detections if d.label == category]
            florence_confirmed = len(category_detections) > 0
            
            if not florence_confirmed:
                frames_without_category += 1
                
                # Try spatial tracking
                tracked_bbox = self.tracker.update(detections)
                
                if tracked_bbox is not None:
                    # Save as uncertain for dream consolidation
                    new_embedding = self.vjepa.encode_region(frames, tracked_bbox)
                    
                    sim_to_original = torch.cosine_similarity(
                        initial_embedding.cpu().flatten().unsqueeze(0),
                        new_embedding.cpu().flatten().unsqueeze(0)
                    ).item()
                    
                    if sim_to_original > 0.25:
                        self.training_collector.record_frame(
                            frames[-1], tracked_bbox, name, category, 
                            confirmed=False
                        )
                        print(f"     📦 Saved uncertain frame (sim: {sim_to_original:.2f}) - will review during dream")
                
                print(f"     ... can't see {category} ({frames_without_category}/{max_frames_without_category})...")
                
                if frames_without_category >= max_frames_without_category:
                    print(f"     Object appears to be removed (Florence can't see {category})")
                    break
                continue
            
            # Florence can see the category - reset counter
            frames_without_category = 0
            
            # Use spatial tracker
            tracked_bbox = self.tracker.update(detections)
            
            if tracked_bbox is None:
                print(f"     Lost spatial tracking...")
                break
            
            # Get embedding
            new_embedding = self.vjepa.encode_region(frames, tracked_bbox)
            
            # Check similarity
            sim_to_last = torch.cosine_similarity(
                last_good_embedding.cpu().flatten().unsqueeze(0),
                new_embedding.cpu().flatten().unsqueeze(0)
            ).item()
            
            sim_to_original = torch.cosine_similarity(
                initial_embedding.cpu().flatten().unsqueeze(0),
                new_embedding.cpu().flatten().unsqueeze(0)
            ).item()
            
            # Accept if similar enough
            if sim_to_last > 0.40 and sim_to_original > 0.30:
                self.memory.add_object(name, category, new_embedding)
                views_captured += 1
                print(f"     View {views_captured} captured (sim: {sim_to_original:.2f}, drift: {sim_to_last:.2f})")
                
                self.training_collector.record_frame(
                    frames[-1], tracked_bbox, name, category,
                    confirmed=True
                )
                
                last_good_embedding = new_embedding.clone()
            else:
                if sim_to_original < 0.30:
                    print(f"     Skipped (drifted too far from original: {sim_to_original:.2f})")
                else:
                    print(f"     Skipped (sudden change: {sim_to_last:.2f})")
        
        self.tracker.stop_tracking()
        print(f"     ✓ Learned '{name}' with {views_captured} views")
        
        # Record learning episode
        self.episodes.record_episode(
            objects=[{
                'name': name,
                'category': category,
                'confidence': 1.0,
                'bbox': initial_bbox,
                'is_focus': True
            }],
            event_type="learning",
            event_detail=f"Learned object '{name}' with {views_captured} views"
        )
    
    def _correct_recognition(self, detection: Detection, embedding: torch.Tensor,
                              frames: np.ndarray, wrong_name: str):
        """Correct a mistaken recognition"""
        print(f"\n  🔧 CORRECTION MODE")
        print(f"     I thought this was: {wrong_name}")
        
        # Check if it's an existing object or new
        known_of_type = self.memory.get_all_of_category(detection.label)
        other_options = [n for n in known_of_type if n != wrong_name]
        
        if other_options:
            print(f"     Other known {detection.label}s:")
            for i, name in enumerate(other_options, 1):
                print(f"       {i}. {name}")
            print(f"       n. It's something new")
            
            response = input("     Which one is it? (number/n): ").strip().lower()
            
            if response == 'n':
                correct_name = self.voice.ask(f"What is this {detection.label} actually called?")
            elif response.isdigit() and 1 <= int(response) <= len(other_options):
                correct_name = other_options[int(response) - 1]
            else:
                print("     Cancelled correction")
                return
        else:
            correct_name = self.voice.ask(f"What is this {detection.label} actually called?")
        
        if len(correct_name) < 2:
            print("     ⚠ Invalid name, cancelled")
            return
        
        # Remove the bad embedding from the wrong object
        print(f"\n     Removing bad view from '{wrong_name}'...")
        if wrong_name in self.memory.objects:
            obj = self.memory.objects[wrong_name]
            if len(obj.embeddings) > 1:
                # Find and remove the most similar embedding (the one that caused the misrecognition)
                max_sim = -1
                max_idx = -1
                for i, stored_emb in enumerate(obj.embeddings):
                    sim = torch.cosine_similarity(
                        embedding.cpu().flatten().unsqueeze(0),
                        stored_emb.flatten().unsqueeze(0)
                    ).item()
                    if sim > max_sim:
                        max_sim = sim
                        max_idx = i
                
                if max_idx >= 0:
                    obj.embeddings.pop(max_idx)
                    print(f"     ✓ Removed confusing view from '{wrong_name}' (was {max_sim:.2f} similar)")
            else:
                print(f"     ⚠ '{wrong_name}' only has 1 view, keeping it")
        
        # Add to correct object (or create it)
        print(f"     Adding view to '{correct_name}'...")
        self.memory.add_object(correct_name, detection.label, embedding)
        
        # Record as training data
        self.training_collector.record_frame(
            frames[-1], detection.bbox, correct_name, detection.label,
            confirmed=True
        )
        
        # Offer to capture more views
        response = input(f"     Want to show me '{correct_name}' from more angles? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            # Re-use learning loop
            self.tracker.start_tracking(detection.bbox, detection.label)
            
            print(f"\n     Show me '{correct_name}' from different angles...")
            print("     (Click debug window and press 'q' to stop early)\n")
            
            views_captured = 1
            max_views = 10
            
            while views_captured < max_views:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 13 or key == 27:
                    break
                
                frames = self._capture_frames(64)
                if frames is None:
                    continue
                
                detections = self.florence.detect(frames[-1])
                
                display = self._draw_debug_frame(frames[-1], detections, None, None)
                cv2.putText(display, f"CORRECTING: {correct_name} ({views_captured}/{max_views})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("JARVIS Debug", display)
                
                tracked_bbox = self.tracker.update(detections)
                
                if tracked_bbox is None:
                    print(f"     Lost sight of object")
                    break
                
                new_embedding = self.vjepa.encode_region(frames, tracked_bbox)
                
                sim = torch.cosine_similarity(
                    embedding.cpu().flatten().unsqueeze(0),
                    new_embedding.cpu().flatten().unsqueeze(0)
                ).item()
                
                if sim > 0.40:
                    self.memory.add_object(correct_name, detection.label, new_embedding)
                    views_captured += 1
                    print(f"     View {views_captured} captured (similarity: {sim:.2f})")
                    
                    self.training_collector.record_frame(
                        frames[-1], tracked_bbox, correct_name, detection.label,
                        confirmed=True
                    )
            
            self.tracker.stop_tracking()
            print(f"     ✓ Updated '{correct_name}' with {views_captured} views")
        
        print(f"\n  ✓ Correction complete: '{wrong_name}' → '{correct_name}'")
        
        # Record correction episode
        self.episodes.record_episode(
            objects=[{
                'name': correct_name,
                'category': detection.label,
                'confidence': 1.0,
                'bbox': detection.bbox,
                'is_focus': True
            }],
            event_type="correction",
            event_detail=f"Corrected misidentification from '{wrong_name}' to '{correct_name}'"
        )
    
    def _shutdown(self):
        """Clean shutdown"""
        print("\n\nShutting down...")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure windows close
        
        self.memory.save()
        self.episodes.save()
        self.episodes.print_summary()
        
        # Save training data for overnight learning
        self.training_collector.save_session()
        
        print(f"\nKnown objects:")
        for name, obj in self.memory.objects.items():
            print(f"  - {name} ({obj.category}, {len(obj.embeddings)} views, seen {obj.times_seen}x)")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    jarvis = WorkshopJARVIS()
    jarvis.run()
