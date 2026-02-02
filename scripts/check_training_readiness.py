"""
Run this before dream_training.py to check if you have enough data
for inward JEPA training.

Usage:
    python check_training_readiness.py
"""

import sys
from pathlib import Path
from collections import Counter

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.episode_memory import EpisodeMemory

def main():
    print("=" * 60)
    print("INWARD JEPA TRAINING READINESS CHECK")
    print("=" * 60)
    
    # Load episode memory
    em = EpisodeMemory(save_path="data/episode_memory.json")
    
    # 1. Episodes with embeddings
    total_episodes = len(em.episodes)
    with_emb = sum(1 for ep in em.episodes if ep.embedding is not None)
    with_objects = sum(1 for ep in em.episodes if ep.object_names)
    with_both = sum(1 for ep in em.episodes if ep.embedding is not None and ep.object_names)
    
    print(f"\n📊 EPISODE COUNTS")
    print(f"   Total episodes: {total_episodes}")
    print(f"   With embeddings: {with_emb}")
    print(f"   With objects: {with_objects}")
    print(f"   With both (usable): {with_both}")
    
    # 2. Object distribution
    all_objects = []
    for ep in em.episodes:
        if ep.embedding is not None:
            all_objects.extend(ep.object_names)
    
    object_counts = Counter(all_objects)
    unique_objects = len(object_counts)
    
    print(f"\n📦 OBJECT DISTRIBUTION")
    print(f"   Unique objects: {unique_objects}")
    print(f"   Total object mentions: {len(all_objects)}")
    print(f"   Top objects:")
    for obj, count in object_counts.most_common(15):
        print(f"      {obj}: {count}")
    
    # 3. Training readiness assessment
    print(f"\n✅ TRAINING READINESS")
    
    # Minimum requirements
    MIN_EPISODES = 10
    MIN_UNIQUE_OBJECTS = 5
    
    ready = True
    issues = []
    
    if with_both < MIN_EPISODES:
        ready = False
        issues.append(f"Not enough usable episodes ({with_both}/{MIN_EPISODES})")
    
    if unique_objects < MIN_UNIQUE_OBJECTS:
        ready = False
        issues.append(f"Not enough unique objects ({unique_objects}/{MIN_UNIQUE_OBJECTS})")
    
    if with_emb == 0:
        ready = False
        issues.append("No episodes have embeddings")
    
    if with_objects == 0:
        ready = False
        issues.append("No episodes have objects")
    
    if ready:
        print("   ✓ Ready for training!")
        print(f"   ✓ You have {with_both} usable episodes")
        print(f"   ✓ You have {unique_objects} unique objects")
    else:
        print("   ✗ NOT ready for training")
        print("   Issues:")
        for issue in issues:
            print(f"      - {issue}")
    
    # 4. Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if not ready:
        print("   To improve training readiness:")
        if with_both < MIN_EPISODES:
            print(f"      - Collect more episodes (need {MIN_EPISODES - with_both} more)")
        if unique_objects < MIN_UNIQUE_OBJECTS:
            print(f"      - Interact with more diverse objects")
        if with_emb == 0:
            print(f"      - Ensure visual embeddings are being captured")
        if with_objects == 0:
            print(f"      - Ensure object detection is working")
        print("   Run active_bernard to collect more episodes!")
    else:
        print("   ✓ You're good to go!")
        print("   Next step: Run dream_training.py to train the model")
        
        # Additional stats for ready systems
        if with_both >= MIN_EPISODES * 2:
            print(f"   ⭐ Excellent! You have {with_both} episodes for robust training")
        
        if unique_objects >= MIN_UNIQUE_OBJECTS * 3:
            print(f"   ⭐ Great diversity! {unique_objects} unique objects")
    
    # 5. Sample episodes
    if with_both > 0:
        print(f"\n📝 SAMPLE EPISODES (showing first 3)")
        sample_count = 0
        for i, ep in enumerate(em.episodes):
            if ep.embedding is not None and ep.object_names and sample_count < 3:
                print(f"\n   Episode {i}:")
                print(f"      Timestamp: {ep.timestamp}")
                print(f"      Objects: {', '.join(ep.object_names[:5])}")
                if len(ep.object_names) > 5:
                    print(f"               ... and {len(ep.object_names) - 5} more")
                print(f"      Embedding shape: {len(ep.embedding) if ep.embedding else 'None'}")
                sample_count += 1
    
    print("\n" + "=" * 60)
    print("Check complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
    
    print("\n" + "=" * 60)
    print("EPISODE DETAIL CHECK")
    print("=" * 60)
    
    em = EpisodeMemory(save_path="data/episode_memory.json")

    for i, ep in enumerate(em.episodes[:5]):  # First 5
        print(f"\nEpisode {i}: {ep.event_type}")
        print(f"  objects list length: {len(ep.objects)}")
        print(f"  object_names: {ep.object_names}")
        if ep.objects:
            for obj in ep.objects[:3]:
                print(f"    - {obj.name} ({obj.category})")
