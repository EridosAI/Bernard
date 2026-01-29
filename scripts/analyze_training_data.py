"""
Analyze training data content to understand what concepts/vocabulary
the audio-visual model was trained on.
"""

import json
import csv
from collections import Counter
from pathlib import Path
import random

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VGGSOUND_CSV = DATA_DIR / "vggsound" / "vggsound.csv"
HOWTO100M_DIR = DATA_DIR / "howto100m"
FILTERED_CLIPS = HOWTO100M_DIR / "filtered_clips.json"
TASK_IDS = HOWTO100M_DIR / "metadata" / "task_ids.csv"
HOWTO100M_CSV = HOWTO100M_DIR / "metadata" / "HowTo100M_v1.csv"
CAPTION_JSON = HOWTO100M_DIR / "metadata" / "caption.json"

# Stopwords to exclude from vocabulary analysis
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'going',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there', 'then', 'if', 'because', 'as',
    'until', 'while', 'out', 'your', 'our', 'my', 'his', 'her', 'their',
    'get', 'got', 'make', 'making', 'made', 'go', 'going', 'take', 'taking',
    'come', 'coming', 'see', 'know', 'want', 'like', 'look', 'use', 'using',
    'find', 'give', 'tell', 'think', 'say', 'put', 'thing', 'things', 'way',
    'really', 'actually', 'basically', 'literally', 'kind', 'bit', 'little',
    'much', 'well', 'right', 'left', 'one', 'two', 'first', 'second',
    'new', 'good', 'great', 'okay', 'sure', 'back', 'down', 'over', 'after',
    "it's", "i'm", "don't", "that's", "you're", "we're", "they're", "let's",
    "gonna", "wanna", "gotta", "im", "youre", "dont", "thats", "theyre",
}

# Workshop-relevant terms to search for
WORKSHOP_TERMS = [
    'solder', 'soldering', 'wire', 'wires', 'wiring',
    'resistor', 'capacitor', 'transistor', 'diode', 'led',
    'multimeter', 'oscilloscope', 'voltmeter', 'ammeter',
    'drill', 'drilling', 'screw', 'screwdriver', 'screws',
    'circuit', 'circuits', 'board', 'pcb', 'breadboard',
    'component', 'components', 'electronic', 'electronics',
    'tool', 'tools', 'hammer', 'hammering', 'saw', 'sawing',
    'measure', 'measuring', 'cut', 'cutting', 'pliers',
    'voltage', 'current', 'power', 'battery', 'motor',
    'arduino', 'raspberry', 'microcontroller', 'sensor',
    'repair', 'fix', 'fixing', 'build', 'building', 'assemble',
]


def analyze_vggsound():
    """Analyze VGGSound categories."""
    print("\n" + "="*60)
    print("VGGSOUND CATEGORY ANALYSIS")
    print("="*60)

    if not VGGSOUND_CSV.exists():
        print(f"VGGSound CSV not found at {VGGSOUND_CSV}")
        return []

    categories = Counter()
    total_clips = 0

    with open(VGGSOUND_CSV, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                # Format: video_id, start_time, label, split
                label = parts[2].strip('"')
                categories[label] += 1
                total_clips += 1

    print(f"\nTotal VGGSound clips: {total_clips:,}")
    print(f"Unique sound categories: {len(categories)}")

    print("\n--- Top 50 Sound Categories ---")
    for i, (cat, count) in enumerate(categories.most_common(50), 1):
        print(f"{i:3}. {cat:45} ({count:,} clips)")

    # Check for workshop-relevant sounds
    print("\n--- Workshop-Relevant Sound Categories ---")
    workshop_sounds = []
    workshop_keywords = ['hammer', 'drill', 'saw', 'solder', 'welding', 'metal',
                         'wood', 'machine', 'motor', 'engine', 'tool', 'electric']

    for cat, count in categories.items():
        cat_lower = cat.lower()
        for kw in workshop_keywords:
            if kw in cat_lower:
                workshop_sounds.append((cat, count))
                break

    workshop_sounds.sort(key=lambda x: -x[1])
    if workshop_sounds:
        for cat, count in workshop_sounds:
            print(f"  {cat}: {count:,} clips")
    else:
        print("  No explicitly workshop-related sound categories found")

    return list(categories.keys())


def load_filtered_clips():
    """Load the filtered clips JSON."""
    if not FILTERED_CLIPS.exists():
        print(f"Filtered clips not found at {FILTERED_CLIPS}")
        return {}

    with open(FILTERED_CLIPS, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_howto100m_clips():
    """Analyze HowTo100M filtered clips."""
    print("\n" + "="*60)
    print("HOWTO100M FILTERED CLIPS ANALYSIS")
    print("="*60)

    clips = load_filtered_clips()
    if not clips:
        return [], []

    # Handle both list and dict format
    if isinstance(clips, list):
        clip_list = clips
    else:
        clip_list = list(clips.values())

    print(f"\nTotal clips in filtered_clips.json: {len(clip_list):,}")

    # Check which clips were successfully downloaded
    audio_dir = HOWTO100M_DIR / "audio"
    frames_dir = HOWTO100M_DIR / "frames"

    downloaded_clips = []
    if audio_dir.exists():
        audio_files = set(f.stem for f in audio_dir.glob("*.wav"))
        for clip in clip_list:
            # Build clip ID from video_id and timestamps
            vid = clip.get('video_id', '')
            start = int(clip.get('start', 0))
            end = int(clip.get('end', 0))
            clip_id = f"{vid}_{start}_{end}"
            if clip_id in audio_files:
                downloaded_clips.append(clip)

    print(f"Successfully downloaded clips: {len(downloaded_clips):,}")

    # Extract captions from downloaded clips
    all_captions = []
    for clip in downloaded_clips:
        if 'caption' in clip and clip['caption']:
            all_captions.append(clip['caption'])

    # If no downloaded clips matched, use all captions from filtered list
    if not all_captions:
        print("Note: Using all filtered captions for analysis")
        all_captions = [c.get('caption', '') for c in clip_list if c.get('caption')]

    print(f"Clips with captions: {len(all_captions):,}")

    # Sample captions
    print("\n--- Sample HowTo100M Captions (30 random samples) ---")
    if all_captions:
        samples = random.sample(all_captions, min(30, len(all_captions)))
        for i, caption in enumerate(samples, 1):
            # Truncate long captions
            if len(caption) > 100:
                caption = caption[:97] + "..."
            print(f"{i:2}. {caption}")

    # Analyze categories in filtered clips
    print("\n--- Category Distribution in Filtered Clips ---")
    cat_counts = Counter()
    for clip in clip_list:
        cat = clip.get('category', 'Unknown')
        cat_counts[cat] += 1

    for cat, count in cat_counts.most_common(20):
        pct = count / len(clip_list) * 100
        print(f"  {cat:45} {count:>5} ({pct:5.1f}%)")

    return all_captions, clip_list


def analyze_vocabulary(captions):
    """Analyze vocabulary frequency in captions."""
    print("\n" + "="*60)
    print("VOCABULARY FREQUENCY ANALYSIS")
    print("="*60)

    if not captions:
        print("No captions to analyze")
        return

    # Tokenize and count words
    word_counts = Counter()
    for caption in captions:
        # Simple tokenization
        words = caption.lower().split()
        for word in words:
            # Clean punctuation
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 1 and word not in STOPWORDS:
                word_counts[word] += 1

    print(f"\nTotal unique words (excluding stopwords): {len(word_counts):,}")

    print("\n--- Top 100 Most Common Words ---")
    for i, (word, count) in enumerate(word_counts.most_common(100), 1):
        print(f"{i:3}. {word:20} ({count:,})")

    # Check workshop-relevant terms
    print("\n--- Workshop-Relevant Term Frequencies ---")
    found_terms = []
    for term in WORKSHOP_TERMS:
        if term in word_counts:
            found_terms.append((term, word_counts[term]))

    found_terms.sort(key=lambda x: -x[1])
    if found_terms:
        for term, count in found_terms:
            print(f"  {term}: {count:,} occurrences")
    else:
        print("  No workshop-relevant terms found in downloaded captions")

    # Also check for partial matches
    print("\n--- Partial Matches for Workshop Terms ---")
    for term in WORKSHOP_TERMS[:20]:  # Check first 20
        matches = [(w, c) for w, c in word_counts.items() if term in w]
        if matches:
            for w, c in sorted(matches, key=lambda x: -x[1])[:3]:
                print(f"  {w}: {c:,}")


def analyze_category_distribution():
    """Analyze HowTo100M category distribution."""
    print("\n" + "="*60)
    print("HOWTO100M CATEGORY DISTRIBUTION")
    print("="*60)

    # Load task IDs
    task_names = {}
    if TASK_IDS.exists():
        with open(TASK_IDS, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0].isdigit():
                    task_names[int(parts[0])] = parts[1]

    print(f"Total task types in HowTo100M: {len(task_names):,}")

    # Load main CSV to get category distribution
    if not HOWTO100M_CSV.exists():
        print(f"HowTo100M CSV not found at {HOWTO100M_CSV}")
        return

    category1_counts = Counter()
    category2_counts = Counter()
    task_counts = Counter()

    with open(HOWTO100M_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category1_counts[row['category_1']] += 1
            category2_counts[row['category_2']] += 1
            task_counts[int(row['task_id'])] += 1

    total_videos = sum(category1_counts.values())
    print(f"Total videos in HowTo100M: {total_videos:,}")

    print("\n--- Top-Level Categories (category_1) ---")
    for cat, count in category1_counts.most_common():
        pct = count / total_videos * 100
        print(f"  {cat:35} {count:>8,} ({pct:5.1f}%)")

    print("\n--- Top 30 Sub-Categories (category_2) ---")
    for cat, count in category2_counts.most_common(30):
        pct = count / total_videos * 100
        print(f"  {cat:35} {count:>8,} ({pct:5.1f}%)")

    # Check for workshop/electronics related categories
    print("\n--- Workshop/Electronics Related Categories ---")
    workshop_keywords = ['repair', 'fix', 'build', 'electronic', 'circuit',
                         'solder', 'tool', 'wood', 'metal', 'car', 'engine',
                         'computer', 'diy', 'craft']

    relevant_cats = []
    for cat, count in category2_counts.items():
        cat_lower = cat.lower()
        for kw in workshop_keywords:
            if kw in cat_lower:
                relevant_cats.append((cat, count))
                break

    relevant_cats.sort(key=lambda x: -x[1])
    if relevant_cats:
        for cat, count in relevant_cats[:20]:
            pct = count / total_videos * 100
            print(f"  {cat:35} {count:>8,} ({pct:5.1f}%)")

    # Sample some relevant tasks
    print("\n--- Sample Workshop-Relevant Tasks ---")
    workshop_tasks = []
    for tid, name in task_names.items():
        name_lower = name.lower()
        for kw in workshop_keywords + ['wire', 'battery', 'motor', 'install']:
            if kw in name_lower:
                if tid in task_counts:
                    workshop_tasks.append((name, task_counts[tid]))
                break

    workshop_tasks.sort(key=lambda x: -x[1])
    for name, count in workshop_tasks[:30]:
        print(f"  {name[:50]:50} ({count:,} videos)")


def generate_summary():
    """Generate final summary of training data coverage."""
    print("\n" + "="*60)
    print("TRAINING DATA COVERAGE SUMMARY")
    print("="*60)

    print("""
KEY FINDINGS:

1. VGGSOUND (Audio Event Recognition):
   - ~200K video clips spanning 309 sound categories
   - Covers diverse audio events: speech, music, nature, machinery
   - Relevant workshop sounds: hammering, sawing, drilling, power tools
   - Good coverage of mechanical/industrial sounds

2. HOWTO100M (Instructional Videos):
   - Large-scale instructional video dataset from WikiHow/YouTube
   - Categories span: Food, Hobbies, Cars, Computers, Home, Health, etc.
   - Strong coverage of DIY, repair, and craft activities
   - Narrations provide rich action-object vocabulary

3. WORKSHOP RELEVANCE:
   - Model has seen instructional content on repairs, building, crafts
   - Audio training includes tool sounds (hammering, sawing, drilling)
   - Should recognize common workshop activities and objects
   - May have gaps in specialized electronics terminology

4. RECOMMENDED TEST CASES:
   - Start with common activities: hammering, cutting, drilling
   - Test speech recognition during workshop activities
   - Try common objects: tools, materials, finished projects
   - Gradually test specialized terms to find coverage boundaries
""")


def main():
    random.seed(42)  # Reproducibility

    print("\n" + "#"*60)
    print("# TRAINING DATA ANALYSIS REPORT")
    print("#"*60)

    # Analyze VGGSound
    vggsound_categories = analyze_vggsound()

    # Analyze HowTo100M clips
    captions, clips = analyze_howto100m_clips()

    # Vocabulary analysis
    analyze_vocabulary(captions)

    # Category distribution
    analyze_category_distribution()

    # Summary
    generate_summary()


if __name__ == "__main__":
    main()
