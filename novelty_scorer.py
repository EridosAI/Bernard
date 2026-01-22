"""
Novelty Scorer - Tracks and scores novelty of scenes and objects
Uses a rolling buffer of recent scene embeddings to determine novelty.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any
import json


class NoveltyScorer:
    """
    Tracks scene history and computes novelty scores for scenes, objects, and positions.
    """
    
    def __init__(self, buffer_size: int = 30, decay_rate: float = 0.05):
        """
        Initialize the NoveltyScorer.
        
        Args:
            buffer_size: Maximum number of scene embeddings to keep in buffer
            decay_rate: Exponential decay rate for weighting recent scenes more heavily
        """
        self.buffer_size = buffer_size
        self.decay_rate = decay_rate
        
        # Rolling buffer of scene embeddings
        self.scene_buffer = deque(maxlen=buffer_size)
        
        # Current scene embedding
        self.current_embedding = None
        
        # Weights for combined novelty calculation
        self.weights = {
            'scene': 0.4,
            'object': 0.3,
            'position': 0.3
        }
    
    def _compute_exponential_weights(self, buffer_length: int) -> np.ndarray:
        """
        Compute exponential decay weights for buffer elements.
        More recent frames get higher weights.
        
        Args:
            buffer_length: Number of elements in buffer
            
        Returns:
            Array of normalized weights
        """
        if buffer_length == 0:
            return np.array([])
        
        # Create weights: most recent = 1.0, older frames decay exponentially
        indices = np.arange(buffer_length)
        weights = np.exp(-self.decay_rate * (buffer_length - 1 - indices))
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1 range, higher = more similar)
        """
        # Flatten embeddings if needed
        emb1 = embedding1.flatten()
        emb2 = embedding2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clip to [0, 1] range (cosine can be -1 to 1, but we use 0 to 1)
        similarity = np.clip((similarity + 1) / 2, 0, 1)
        
        return float(similarity)
    
    def update_scene(self, embedding: np.ndarray) -> None:
        """
        Add current scene embedding to the buffer.
        
        Args:
            embedding: Scene embedding from V-JEPA (full-frame encoding)
        """
        # Store current embedding
        self.current_embedding = embedding.copy()
        
        # Add to buffer (automatically removes oldest if at max size)
        self.scene_buffer.append(embedding.copy())
    
    def get_scene_novelty(self) -> float:
        """
        Compute novelty of current scene compared to recent history.
        
        Returns:
            Novelty score (0-1), where 1 = completely novel, 0 = identical to history
        """
        if self.current_embedding is None:
            return 0.0
        
        if len(self.scene_buffer) <= 1:
            # First scene is maximally novel
            return 1.0
        
        # Get all buffer embeddings except the current one
        buffer_embeddings = list(self.scene_buffer)[:-1]
        
        # Compute weighted similarities to all previous scenes
        weights = self._compute_exponential_weights(len(buffer_embeddings))
        
        # Compute similarity to each previous scene
        similarities = []
        for prev_embedding in buffer_embeddings:
            sim = self._cosine_similarity(self.current_embedding, prev_embedding)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Compute weighted maximum similarity
        # Recent scenes matter more due to exponential weights
        max_similarity = np.max(similarities * weights) / np.max(weights)
        
        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity
        
        return float(np.clip(novelty, 0, 1))
    
    def get_object_novelty(self, confidence: float) -> float:
        """
        Compute object novelty based on recognition confidence.
        Low confidence = high novelty (object is unfamiliar).
        
        Args:
            confidence: Recognition confidence score (0-1)
            
        Returns:
            Novelty score (0-1), where 1 = completely novel, 0 = highly confident recognition
        """
        # Simple inverse of confidence
        novelty = 1.0 - confidence
        
        return float(np.clip(novelty, 0, 1))
    
    def get_position_novelty(
        self,
        object_name: str,
        position: Tuple[float, float],
        episode_memory: Any
    ) -> float:
        """
        Compute position novelty - how far is object from its typical position.
        
        Args:
            object_name: Name of the object
            position: Current position (x, y) normalized to [0, 1]
            episode_memory: Episode memory object with object history
            
        Returns:
            Novelty score (0-1), where 1 = very far from typical, 0 = at typical position
        """
        try:
            # Get object history from episode memory
            if hasattr(episode_memory, 'memory') and 'objects' in episode_memory.memory:
                objects = episode_memory.memory['objects']
                
                # Find this object in memory
                if object_name not in objects:
                    # Object never seen before = maximally novel position
                    return 1.0
                
                obj_data = objects[object_name]
                positions = obj_data.get('positions', [])
                
                if len(positions) == 0:
                    return 1.0
                
                # Calculate typical position (mean)
                positions_array = np.array(positions)
                typical_position = np.mean(positions_array, axis=0)
                
                # Calculate distance from typical position
                current_pos = np.array(position)
                distance = np.linalg.norm(current_pos - typical_position)
                
                # Normalize distance (diagonal of unit square = sqrt(2))
                # Scale so max distance = 1.0
                max_distance = np.sqrt(2)
                normalized_distance = distance / max_distance
                
                # Apply non-linear scaling to make small deviations less novel
                # Use square root to emphasize larger deviations
                novelty = np.sqrt(normalized_distance)
                
                return float(np.clip(novelty, 0, 1))
            
            else:
                # No memory available, assume novel
                return 1.0
                
        except Exception as e:
            # If any error occurs, return moderate novelty
            print(f"Error computing position novelty: {e}")
            return 0.5
    
    def get_combined_novelty(
        self,
        object_name: str,
        confidence: float,
        position: Tuple[float, float],
        episode_memory: Any
    ) -> Dict[str, float]:
        """
        Compute weighted combination of all novelty metrics.
        
        Args:
            object_name: Name of the object
            confidence: Recognition confidence score (0-1)
            position: Current position (x, y) normalized to [0, 1]
            episode_memory: Episode memory object with object history
            
        Returns:
            Dictionary containing individual and combined novelty scores
        """
        # Compute individual novelty scores
        scene_novelty = self.get_scene_novelty()
        object_novelty = self.get_object_novelty(confidence)
        position_novelty = self.get_position_novelty(object_name, position, episode_memory)
        
        # Compute weighted combination
        combined_novelty = (
            self.weights['scene'] * scene_novelty +
            self.weights['object'] * object_novelty +
            self.weights['position'] * position_novelty
        )
        
        return {
            'scene_novelty': float(scene_novelty),
            'object_novelty': float(object_novelty),
            'position_novelty': float(position_novelty),
            'combined_novelty': float(combined_novelty),
            'weights': self.weights.copy()
        }
    
    def set_weights(self, scene: float = 0.4, object: float = 0.3, position: float = 0.3) -> None:
        """
        Set custom weights for combined novelty calculation.
        
        Args:
            scene: Weight for scene novelty (0-1)
            object: Weight for object novelty (0-1)
            position: Weight for position novelty (0-1)
        """
        # Normalize weights to sum to 1
        total = scene + object + position
        if total == 0:
            total = 1.0
        
        self.weights = {
            'scene': scene / total,
            'object': object / total,
            'position': position / total
        }
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current buffer state.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'buffer_size': len(self.scene_buffer),
            'max_buffer_size': self.buffer_size,
            'buffer_full': len(self.scene_buffer) == self.buffer_size,
            'decay_rate': self.decay_rate,
            'has_current_scene': self.current_embedding is not None
        }
    
    def reset(self) -> None:
        """
        Reset the novelty scorer, clearing all buffers.
        """
        self.scene_buffer.clear()
        self.current_embedding = None
    
    def __repr__(self) -> str:
        return (f"NoveltyScorer(buffer_size={self.buffer_size}, "
                f"current_scenes={len(self.scene_buffer)}, "
                f"decay_rate={self.decay_rate})")


def main():
    """
    Example usage and testing of NoveltyScorer.
    """
    print("=== Novelty Scorer Test ===\n")
    
    # Initialize scorer
    scorer = NoveltyScorer(buffer_size=30, decay_rate=0.05)
    print(f"Created: {scorer}\n")
    
    # Simulate some scene embeddings
    print("Simulating scene updates...")
    
    # First scene - should be maximally novel
    scene1 = np.random.randn(512)
    scorer.update_scene(scene1)
    novelty1 = scorer.get_scene_novelty()
    print(f"Scene 1 novelty: {novelty1:.3f} (should be 1.0 - first scene)")
    
    # Very similar scene - should have low novelty
    scene2 = scene1 + np.random.randn(512) * 0.01
    scorer.update_scene(scene2)
    novelty2 = scorer.get_scene_novelty()
    print(f"Scene 2 novelty: {novelty2:.3f} (similar to scene 1)")
    
    # Very different scene - should have high novelty
    scene3 = np.random.randn(512) * 2
    scorer.update_scene(scene3)
    novelty3 = scorer.get_scene_novelty()
    print(f"Scene 3 novelty: {novelty3:.3f} (very different)\n")
    
    # Test object novelty
    print("Testing object novelty (inverse of confidence)...")
    print(f"High confidence (0.9): novelty = {scorer.get_object_novelty(0.9):.3f}")
    print(f"Medium confidence (0.5): novelty = {scorer.get_object_novelty(0.5):.3f}")
    print(f"Low confidence (0.1): novelty = {scorer.get_object_novelty(0.1):.3f}\n")
    
    # Test combined novelty
    print("Testing combined novelty...")
    
    # Create a mock episode memory
    class MockEpisodeMemory:
        def __init__(self):
            self.memory = {
                'objects': {
                    'bottle': {
                        'positions': [(0.5, 0.5), (0.52, 0.48), (0.48, 0.52)]
                    }
                }
            }
    
    mock_memory = MockEpisodeMemory()
    
    # Test object at typical position
    result1 = scorer.get_combined_novelty(
        'bottle', 
        confidence=0.8, 
        position=(0.5, 0.5), 
        episode_memory=mock_memory
    )
    print(f"Bottle at typical position (0.5, 0.5):")
    print(f"  Combined novelty: {result1['combined_novelty']:.3f}")
    print(f"  Components: scene={result1['scene_novelty']:.3f}, "
          f"object={result1['object_novelty']:.3f}, "
          f"position={result1['position_novelty']:.3f}\n")
    
    # Test object at unusual position
    result2 = scorer.get_combined_novelty(
        'bottle',
        confidence=0.8,
        position=(0.9, 0.1),
        episode_memory=mock_memory
    )
    print(f"Bottle at unusual position (0.9, 0.1):")
    print(f"  Combined novelty: {result2['combined_novelty']:.3f}")
    print(f"  Components: scene={result2['scene_novelty']:.3f}, "
          f"object={result2['object_novelty']:.3f}, "
          f"position={result2['position_novelty']:.3f}\n")
    
    # Test new object
    result3 = scorer.get_combined_novelty(
        'unknown_object',
        confidence=0.3,
        position=(0.7, 0.3),
        episode_memory=mock_memory
    )
    print(f"Unknown object at (0.7, 0.3):")
    print(f"  Combined novelty: {result3['combined_novelty']:.3f}")
    print(f"  Components: scene={result3['scene_novelty']:.3f}, "
          f"object={result3['object_novelty']:.3f}, "
          f"position={result3['position_novelty']:.3f}\n")
    
    # Show buffer stats
    stats = scorer.get_buffer_stats()
    print(f"Buffer stats: {stats}")


if __name__ == "__main__":
    main()
