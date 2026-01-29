# episode_memory.py - Episode Memory System for Bernard
"""
Stores experiences as episodes, not just objects.
Foundation for associative memory and causal learning.

An episode captures:
- What objects were present
- Where they were (spatial relationships)
- When it happened
- What was happening (event type)
"""

import json
import time
import os
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict


@dataclass
class SpatialInfo:
    """Spatial information about an object in a frame"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]          # center point
    area: int                         # bbox area (proxy for distance/size)
    
    @classmethod
    def from_bbox(cls, bbox: Tuple[int, int, int, int]) -> 'SpatialInfo':
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        area = (x2 - x1) * (y2 - y1)
        return cls(bbox=bbox, center=center, area=area)


@dataclass
class ObjectSighting:
    """A single object sighting within an episode"""
    name: str                    # Your name: "air conditioner remote"
    category: str                # Florence category: "remote control"
    confidence: float            # Recognition confidence
    spatial: SpatialInfo         # Where in frame
    is_focus: bool = False       # Was this the focus object?


@dataclass 
class Episode:
    """A single moment/experience"""
    timestamp: float
    objects: List[ObjectSighting]
    event_type: str              # "observation", "learning", "correction", "interaction"
    event_detail: str = ""       # Additional context
    
    # V-JEPA embedding of the frame
    embedding: Optional[np.ndarray] = None
    
    # Derived on save
    object_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.object_names = [obj.name for obj in self.objects if obj.name]


@dataclass
class CoOccurrence:
    """Tracks how often two objects appear together"""
    object_a: str
    object_b: str
    count: int = 0
    last_seen: float = 0.0
    
    # Spatial relationship stats
    a_left_of_b: int = 0
    a_above_b: int = 0
    a_near_b: int = 0  # Centers within threshold


class EpisodeMemory:
    """
    Stores and queries episodic memories.
    
    Enables:
    - "What objects appear together?" (co-occurrence)
    - "Where is X usually located?" (spatial priors)
    - "What happened last time I saw X?" (temporal queries)
    """
    
    def __init__(self, save_path: str = "data/episode_memory.json"):
        self.save_path = save_path
        self.episodes: List[Episode] = []
        self.co_occurrences: Dict[str, CoOccurrence] = {}  # "obj_a|obj_b" -> CoOccurrence
        
        # Spatial priors: where objects tend to appear
        self.spatial_priors: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        
        # Load existing data
        self._load()
    
    def record_episode(self, 
                       objects: List[Dict],  # [{name, category, confidence, bbox, is_focus}]
                       event_type: str = "observation",
                       event_detail: str = "",
                       embedding: Optional[np.ndarray] = None) -> Episode:
        """
        Record a new episode.
        
        Args:
            objects: List of dicts with object info
            event_type: "observation", "learning", "correction", "interaction"
            event_detail: Additional context string
            embedding: Optional V-JEPA embedding of the frame
        """
        # Build object sightings
        sightings = []
        for obj in objects:
            if obj.get('bbox'):
                spatial = SpatialInfo.from_bbox(tuple(obj['bbox']))
            else:
                spatial = SpatialInfo(bbox=(0,0,0,0), center=(0,0), area=0)
            
            sighting = ObjectSighting(
                name=obj.get('name', ''),
                category=obj.get('category', ''),
                confidence=obj.get('confidence', 0.0),
                spatial=spatial,
                is_focus=obj.get('is_focus', False)
            )
            sightings.append(sighting)
        
        # Create episode
        episode = Episode(
            timestamp=time.time(),
            objects=sightings,
            event_type=event_type,
            event_detail=event_detail,
            embedding=embedding
        )
        
        self.episodes.append(episode)
        
        # Update co-occurrences
        self._update_co_occurrences(sightings)
        
        # Update spatial priors
        self._update_spatial_priors(sightings)
        
        return episode
    
    def _update_co_occurrences(self, sightings: List[ObjectSighting]):
        """Update co-occurrence counts for all object pairs"""
        # Get unique named objects
        named = [s for s in sightings if s.name]
        
        for i, obj_a in enumerate(named):
            for obj_b in named[i+1:]:
                # Create canonical key (alphabetical order)
                key = self._co_key(obj_a.name, obj_b.name)
                
                if key not in self.co_occurrences:
                    names = sorted([obj_a.name, obj_b.name])
                    self.co_occurrences[key] = CoOccurrence(
                        object_a=names[0],
                        object_b=names[1]
                    )
                
                co = self.co_occurrences[key]
                co.count += 1
                co.last_seen = time.time()
                
                # Update spatial relationships
                # Determine which is a and which is b based on key order
                if obj_a.name < obj_b.name:
                    a, b = obj_a, obj_b
                else:
                    a, b = obj_b, obj_a
                
                # Left/right
                if a.spatial.center[0] < b.spatial.center[0]:
                    co.a_left_of_b += 1
                
                # Above/below
                if a.spatial.center[1] < b.spatial.center[1]:
                    co.a_above_b += 1
                
                # Near (centers within 200px)
                dist = ((a.spatial.center[0] - b.spatial.center[0])**2 + 
                        (a.spatial.center[1] - b.spatial.center[1])**2) ** 0.5
                if dist < 200:
                    co.a_near_b += 1
    
    def _update_spatial_priors(self, sightings: List[ObjectSighting]):
        """Track where objects tend to appear"""
        for s in sightings:
            if s.name:
                self.spatial_priors[s.name].append(s.spatial.center)
                # Keep only last 100 positions
                if len(self.spatial_priors[s.name]) > 100:
                    self.spatial_priors[s.name] = self.spatial_priors[s.name][-100:]
    
    @staticmethod
    def _co_key(name_a: str, name_b: str) -> str:
        """Create canonical key for co-occurrence lookup"""
        return "|".join(sorted([name_a, name_b]))
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_co_occurring_objects(self, object_name: str, min_count: int = 2) -> List[Tuple[str, int]]:
        """
        Get objects that frequently appear with the given object.
        Returns: [(other_object_name, count), ...] sorted by count descending
        """
        results = []
        
        for key, co in self.co_occurrences.items():
            if co.count >= min_count:
                if co.object_a == object_name:
                    results.append((co.object_b, co.count))
                elif co.object_b == object_name:
                    results.append((co.object_a, co.count))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_spatial_prior(self, object_name: str) -> Optional[Tuple[int, int]]:
        """
        Get the typical location for an object (average of past positions).
        Returns: (x, y) or None if not enough data
        """
        positions = self.spatial_priors.get(object_name, [])
        
        if len(positions) < 3:
            return None
        
        avg_x = sum(p[0] for p in positions) // len(positions)
        avg_y = sum(p[1] for p in positions) // len(positions)
        
        return (avg_x, avg_y)
    
    def get_recent_episodes(self, 
                            object_name: Optional[str] = None,
                            event_type: Optional[str] = None,
                            limit: int = 10) -> List[Episode]:
        """
        Get recent episodes, optionally filtered.
        """
        results = self.episodes
        
        if object_name:
            results = [e for e in results if object_name in e.object_names]
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        
        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_last_seen(self, object_name: str) -> Optional[Episode]:
        """Get the most recent episode containing this object"""
        for episode in reversed(self.episodes):
            if object_name in episode.object_names:
                return episode
        return None
    
    def get_objects_seen_with(self, object_name: str) -> Set[str]:
        """Get all objects ever seen with this object"""
        seen_with = set()
        
        for episode in self.episodes:
            if object_name in episode.object_names:
                seen_with.update(episode.object_names)
        
        seen_with.discard(object_name)  # Don't include self
        return seen_with
    
    def sample_training_batch(self, 
                          batch_size: int = 32,
                          num_negatives: int = 7) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample (anchor, positive, negatives) tuples for InwardJEPA training.
        
        Positive pairs: episodes sharing at least one object
        Negatives: episodes with no object overlap with anchor
        
        Returns:
            anchors: [B, D]
            positives: [B, D]
            negatives: [B, N, D]
            
            Or None if not enough episodes with embeddings
        """
        import random
        
        # Get episodes that have embeddings and objects
        episodes_with_emb = [(i, ep) for i, ep in enumerate(self.episodes) 
                             if ep.embedding is not None and ep.object_names]
        
        if len(episodes_with_emb) < batch_size + num_negatives:
            return None
        
        anchors = []
        positives = []
        negatives_list = []
        
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(anchors) < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Pick random anchor
            anchor_idx, anchor_ep = random.choice(episodes_with_emb)
            anchor_objects = set(
                obj.name for obj in anchor_ep.objects 
                if obj.is_focus and obj.name
            )
            
            # Find positive: shares at least one object
            positive_candidates = [
                (i, ep) for i, ep in episodes_with_emb
                if i != anchor_idx and 
                   anchor_objects.intersection(
                       obj.name for obj in ep.objects if obj.is_focus and obj.name
                   )
            ]
            
            if not positive_candidates:
                continue
            
            # Find negatives: no object overlap
            negative_candidates = [
                (i, ep) for i, ep in episodes_with_emb
                if i != anchor_idx and
                   not anchor_objects.intersection(
                       obj.name for obj in ep.objects if obj.is_focus and obj.name
                   )
            ]
            
            if len(negative_candidates) < num_negatives:
                continue
            
            # Sample
            pos_idx, pos_ep = random.choice(positive_candidates)
            neg_samples = random.sample(negative_candidates, num_negatives)
            
            anchors.append(anchor_ep.embedding)
            positives.append(pos_ep.embedding)
            negatives_list.append([ep.embedding for _, ep in neg_samples])
        
        if len(anchors) < batch_size:
            return None
        
        return (
            np.stack(anchors),
            np.stack(positives),
            np.stack([np.stack(negs) for negs in negatives_list])
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self):
        """Save episodes and co-occurrences to disk"""
        data = {
            'episodes': [],
            'co_occurrences': {},
            'spatial_priors': dict(self.spatial_priors)
        }
        
        # Serialize episodes
        for ep in self.episodes:
            ep_dict = {
                'timestamp': ep.timestamp,
                'event_type': ep.event_type,
                'event_detail': ep.event_detail,
                'object_names': ep.object_names,
                'objects': []
            }
            for obj in ep.objects:
                ep_dict['objects'].append({
                    'name': obj.name,
                    'category': obj.category,
                    'confidence': obj.confidence,
                    'bbox': obj.spatial.bbox,
                    'center': obj.spatial.center,
                    'area': obj.spatial.area,
                    'is_focus': obj.is_focus
                })
            data['episodes'].append(ep_dict)
        
        # Serialize co-occurrences
        for key, co in self.co_occurrences.items():
            data['co_occurrences'][key] = {
                'object_a': co.object_a,
                'object_b': co.object_b,
                'count': co.count,
                'last_seen': co.last_seen,
                'a_left_of_b': co.a_left_of_b,
                'a_above_b': co.a_above_b,
                'a_near_b': co.a_near_b
            }
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings separately
        embeddings = {}
        for i, ep in enumerate(self.episodes):
            if ep.embedding is not None:
                embeddings[str(i)] = ep.embedding
        
        if embeddings:
            np.savez(self.save_path.replace('.json', '_embeddings.npz'), **embeddings)
    
    def _load(self):
        """Load episodes from disk"""
        if not os.path.exists(self.save_path):
            return
        
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            # Load episodes
            for ep_dict in data.get('episodes', []):
                sightings = []
                for obj in ep_dict.get('objects', []):
                    spatial = SpatialInfo(
                        bbox=tuple(obj['bbox']),
                        center=tuple(obj['center']),
                        area=obj['area']
                    )
                    sighting = ObjectSighting(
                        name=obj['name'],
                        category=obj['category'],
                        confidence=obj['confidence'],
                        spatial=spatial,
                        is_focus=obj.get('is_focus', False)
                    )
                    sightings.append(sighting)
                
                episode = Episode(
                    timestamp=ep_dict['timestamp'],
                    objects=sightings,
                    event_type=ep_dict['event_type'],
                    event_detail=ep_dict.get('event_detail', '')
                )
                self.episodes.append(episode)
            
            # Load co-occurrences
            for key, co_dict in data.get('co_occurrences', {}).items():
                self.co_occurrences[key] = CoOccurrence(
                    object_a=co_dict['object_a'],
                    object_b=co_dict['object_b'],
                    count=co_dict['count'],
                    last_seen=co_dict['last_seen'],
                    a_left_of_b=co_dict.get('a_left_of_b', 0),
                    a_above_b=co_dict.get('a_above_b', 0),
                    a_near_b=co_dict.get('a_near_b', 0)
                )
            
            # Load spatial priors
            for name, positions in data.get('spatial_priors', {}).items():
                self.spatial_priors[name] = [tuple(p) for p in positions]
            
            # Load embeddings separately
            embeddings_path = self.save_path.replace('.json', '_embeddings.npz')
            if os.path.exists(embeddings_path):
                embeddings_data = np.load(embeddings_path)
                for i, episode in enumerate(self.episodes):
                    key = str(i)
                    if key in embeddings_data:
                        episode.embedding = embeddings_data[key]
            
            print(f"  Loaded {len(self.episodes)} episodes, {len(self.co_occurrences)} co-occurrences")
            
        except Exception as e:
            print(f"  Warning: Could not load episode memory: {e}")
    
    # =========================================================================
    # STATS / DEBUG
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_episodes': len(self.episodes),
            'unique_objects': len(self.spatial_priors),
            'co_occurrence_pairs': len(self.co_occurrences),
            'episodes_by_type': self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count episodes by event type"""
        counts = defaultdict(int)
        for ep in self.episodes:
            counts[ep.event_type] += 1
        return dict(counts)
    
    def print_summary(self):
        """Print a summary of episodic memory"""
        stats = self.get_stats()
        
        print(f"\n📚 Episode Memory Summary:")
        print(f"   Total episodes: {stats['total_episodes']}")
        print(f"   Unique objects: {stats['unique_objects']}")
        print(f"   Co-occurrence pairs: {stats['co_occurrence_pairs']}")
        
        if stats['episodes_by_type']:
            print(f"   By type:")
            for event_type, count in stats['episodes_by_type'].items():
                print(f"     - {event_type}: {count}")
        
        # Top co-occurrences
        if self.co_occurrences:
            print(f"\n   Top co-occurrences:")
            sorted_co = sorted(self.co_occurrences.values(), 
                              key=lambda c: c.count, reverse=True)[:5]
            for co in sorted_co:
                print(f"     - {co.object_a} + {co.object_b}: {co.count}x")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test
    em = EpisodeMemory(save_path="test_episode_memory.json")
    
    # Simulate some episodes
    em.record_episode(
        objects=[
            {'name': 'scissors', 'category': 'scissors', 'confidence': 0.95, 
             'bbox': (100, 100, 200, 200), 'is_focus': True},
            {'name': 'tape', 'category': 'tape', 'confidence': 0.88,
             'bbox': (250, 100, 350, 180), 'is_focus': False},
        ],
        event_type="observation"
    )
    
    em.record_episode(
        objects=[
            {'name': 'scissors', 'category': 'scissors', 'confidence': 0.92,
             'bbox': (110, 105, 210, 205), 'is_focus': False},
            {'name': 'tape', 'category': 'tape', 'confidence': 0.90,
             'bbox': (240, 95, 340, 175), 'is_focus': False},
            {'name': 'remote', 'category': 'remote control', 'confidence': 0.87,
             'bbox': (400, 200, 480, 320), 'is_focus': True},
        ],
        event_type="observation"
    )
    
    em.record_episode(
        objects=[
            {'name': 'scissors', 'category': 'scissors', 'confidence': 0.94,
             'bbox': (105, 102, 205, 202), 'is_focus': True},
        ],
        event_type="learning",
        event_detail="User showed scissors from new angle"
    )
    
    # Test queries
    print("\n--- Testing Queries ---")
    
    print(f"\nObjects seen with scissors: {em.get_objects_seen_with('scissors')}")
    print(f"Co-occurring with scissors: {em.get_co_occurring_objects('scissors', min_count=1)}")
    print(f"Spatial prior for scissors: {em.get_spatial_prior('scissors')}")
    
    # Save and reload
    em.save()
    em.print_summary()
    
    print("\n--- Reloading ---")
    em2 = EpisodeMemory(save_path="test_episode_memory.json")
    em2.print_summary()
    
    # Cleanup test file
    os.remove("test_episode_memory.json")
    print("\n✓ Test complete!")
