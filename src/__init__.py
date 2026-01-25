# Arnold - Core modules
"""
Core modules for the Arnold workshop assistant.

Arnold = Associative Recognition Network for Observation-Linked Development

Main entry points:
    - arnold_integrated_v2: Main orchestration loop
    - dream_training: Overnight LoRA training
    - workshop_session: Interactive teaching sessions

Supporting modules:
    - episode_memory: Experience storage and retrieval
    - novelty_scorer: Curiosity-based attention
    - continuous_listening: Voice interaction pipeline
    - inward_jepa: Associative memory predictor
    - beats_encoder: Audio embedding
"""

from .episode_memory import EpisodeMemory
from .novelty_scorer import NoveltyScorer
from .continuous_listening import ContinuousListener, Intent, VoiceEvent

__all__ = [
    'EpisodeMemory',
    'NoveltyScorer',
    'ContinuousListener',
    'Intent',
    'VoiceEvent',
]
