"""
HowTo100M Dataset Downloader

Downloads a filtered subset of HowTo100M for audio-visual contrastive learning.
Focuses on workshop-relevant categories: DIY, repair, crafts, cooking.

Setup:
    1. Download "All-in-One zip" from https://www.di.ens.fr/willow/research/howto100m/
    2. Extract to data/howto100m/metadata/
    3. Run this script

Expected files in data/howto100m/metadata/:
    - HowTo100M_v1.csv (video_id, start, end, caption)
    - task_ids.csv (maps task_id to category hierarchy)
    - possibly other metadata files
"""

import os
import json
import subprocess
import random
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from tqdm import tqdm
import tempfile


@dataclass
class HowTo100MConfig:
    """Configuration for HowTo100M download"""
    data_dir: str = "./data/howto100m"
    metadata_dir: str = "./data/howto100m/metadata"

    # Target clip count
    target_clips: int = 10000
    max_clips_per_video: int = 5  # Diversity

    # Clip duration filter (seconds)
    min_duration: float = 3.0
    max_duration: float = 10.0

    # Audio/video settings
    audio_sample_rate: int = 16000
    frame_size: int = 256

    # Workshop-relevant category filters (partial matches)
    target_categories: List[str] = field(default_factory=lambda: [
        # DIY and Home Improvement
        "Home and Garden",
        "DIY",
        "home improvement",
        "woodworking",
        "carpentry",
        "plumbing",
        "electrical",

        # Repair
        "repair",
        "fix",
        "Cars & Other Vehicles",
        "Auto Repair",
        "maintenance",

        # Crafts and Hobbies
        "Hobbies and Crafts",
        "craft",
        "sewing",
        "knitting",

        # Electronics
        "Computers and Electronics",
        "electronics",
        "soldering",

        # Cooking (lots of visible actions with narration)
        "Food and Entertaining",
        "Cooking",
        "recipe",
    ])


class HowTo100MDownloader:
    """Downloads filtered HowTo100M clips"""

    def __init__(self, config: HowTo100MConfig = None):
        self.config = config or HowTo100MConfig()
        self.data_dir = Path(self.config.data_dir)
        self.metadata_dir = Path(self.config.metadata_dir)
        self.audio_dir = self.data_dir / "audio"
        self.frames_dir = self.data_dir / "frames"
        self.log_path = self.data_dir / "download_log.json"
        self.filtered_clips_path = self.data_dir / "filtered_clips.json"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

    def find_metadata_files(self) -> Dict[str, Path]:
        """Find HowTo100M metadata files"""
        files = {}

        # Look for main CSV (various possible names)
        for pattern in ["HowTo100M*.csv", "howto100m*.csv", "*.csv"]:
            matches = list(self.metadata_dir.glob(pattern))
            if matches:
                # Prefer the main data file (largest or with v1 in name)
                for m in matches:
                    if "v1" in m.name.lower() or m.stat().st_size > 1_000_000:
                        files['main'] = m
                        break
                if 'main' not in files and matches:
                    files['main'] = matches[0]

        # Look for task/category mapping
        for pattern in ["task*.csv", "category*.csv"]:
            matches = list(self.metadata_dir.glob(pattern))
            if matches:
                files['tasks'] = matches[0]

        return files

    def load_task_categories(self, task_file: Path) -> Dict[str, str]:
        """Load task ID to task name mapping (tab-separated: id\tname)"""
        task_to_name = {}

        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        task_id = parts[0]
                        task_name = parts[1]
                        task_to_name[task_id] = task_name
        except Exception as e:
            print(f"Warning: Could not load task categories: {e}")

        return task_to_name

    def matches_target_category(self, category: str) -> bool:
        """Check if category matches our target categories"""
        category_lower = category.lower()
        for target in self.config.target_categories:
            if target.lower() in category_lower:
                return True
        return False

    def load_and_filter_clips(self) -> List[Dict]:
        """Load HowTo100M data and filter to relevant clips"""
        files = self.find_metadata_files()

        if 'main' not in files:
            raise FileNotFoundError(
                f"HowTo100M CSV not found in {self.metadata_dir}\n"
                f"Download from: https://www.di.ens.fr/willow/research/howto100m/"
            )

        # Check for caption.json
        caption_file = self.metadata_dir / "caption.json"
        if not caption_file.exists():
            raise FileNotFoundError(
                f"caption.json not found in {self.metadata_dir}\n"
                f"This file contains the clip timestamps and captions."
            )

        # Step 1: Load video -> category mapping from HowTo100M_v1.csv
        print(f"Loading video categories from {files['main']}...")
        video_categories = {}  # video_id -> (category_1, category_2)

        with open(files['main'], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get('video_id')
                if video_id:
                    cat1 = row.get('category_1', '')
                    cat2 = row.get('category_2', '')
                    video_categories[video_id] = (cat1, cat2)

        print(f"  Loaded {len(video_categories)} videos")

        # Step 2: Filter videos by category
        filtered_videos = set()
        for video_id, (cat1, cat2) in video_categories.items():
            if self.matches_target_category(cat1) or self.matches_target_category(cat2):
                filtered_videos.add(video_id)

        print(f"  {len(filtered_videos)} videos match target categories")

        # Step 3: Load clips from caption.json for filtered videos
        print(f"Loading clips from caption.json (this may take a moment)...")
        all_clips = []
        video_clip_counts = {}

        import ijson  # Streaming JSON parser

        try:
            # Try streaming parser for large JSON
            with open(caption_file, 'rb') as f:
                parser = ijson.kvitems(f, '')
                for video_id, data in parser:
                    if video_id not in filtered_videos:
                        continue

                    starts = data.get('start', [])
                    ends = data.get('end', [])
                    texts = data.get('text', [])

                    cat1, cat2 = video_categories.get(video_id, ('', ''))
                    category = f"{cat1} > {cat2}" if cat2 else cat1

                    for i, (start, end) in enumerate(zip(starts, ends)):
                        duration = end - start
                        if duration < self.config.min_duration or duration > self.config.max_duration:
                            continue

                        if video_clip_counts.get(video_id, 0) >= self.config.max_clips_per_video:
                            break

                        video_clip_counts[video_id] = video_clip_counts.get(video_id, 0) + 1

                        caption = texts[i] if i < len(texts) else ''

                        all_clips.append({
                            'video_id': video_id,
                            'start': float(start),
                            'end': float(end),
                            'duration': float(duration),
                            'category': category,
                            'caption': str(caption)[:200]
                        })

                        if len(all_clips) >= self.config.target_clips * 2:
                            # Collected enough, will sample later
                            break

                    if len(all_clips) >= self.config.target_clips * 2:
                        break

        except ImportError:
            # Fallback: load entire JSON (needs ~8GB RAM)
            print("  Note: Install 'ijson' for memory-efficient parsing: pip install ijson")
            print("  Loading entire caption.json into memory...")

            import json
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)

            for video_id in filtered_videos:
                if video_id not in captions:
                    continue

                data = captions[video_id]
                starts = data.get('start', [])
                ends = data.get('end', [])
                texts = data.get('text', [])

                cat1, cat2 = video_categories.get(video_id, ('', ''))
                category = f"{cat1} > {cat2}" if cat2 else cat1

                for i, (start, end) in enumerate(zip(starts, ends)):
                    duration = end - start
                    if duration < self.config.min_duration or duration > self.config.max_duration:
                        continue

                    if video_clip_counts.get(video_id, 0) >= self.config.max_clips_per_video:
                        break

                    video_clip_counts[video_id] = video_clip_counts.get(video_id, 0) + 1

                    caption = texts[i] if i < len(texts) else ''

                    all_clips.append({
                        'video_id': video_id,
                        'start': start,
                        'end': end,
                        'duration': duration,
                        'category': category,
                        'caption': caption[:200]
                    })

        print(f"  Found {len(all_clips)} clips matching criteria")

        # Random sample to target count
        if len(all_clips) > self.config.target_clips:
            all_clips = random.sample(all_clips, self.config.target_clips)
            print(f"  Sampled {len(all_clips)} clips")

        return all_clips

    def _get_clip_id(self, clip: Dict) -> str:
        """Generate unique clip ID"""
        return f"{clip['video_id']}_{int(clip['start'])}_{int(clip['end'])}"

    def _load_progress(self) -> Dict:
        """Load download progress"""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_progress(self, log: Dict):
        """Save download progress"""
        with open(self.log_path, 'w') as f:
            json.dump(log, f, indent=2)

    def download_clip(self, clip: Dict) -> Optional[Dict]:
        """Download single clip, extract audio and frame"""
        video_id = clip['video_id']
        start = clip['start']
        end = clip['end']
        clip_id = self._get_clip_id(clip)

        audio_path = self.audio_dir / f"{clip_id}.wav"
        frame_path = self.frames_dir / f"{clip_id}.jpg"

        # Skip if already exists
        if audio_path.exists() and frame_path.exists():
            return {
                'audio_path': str(audio_path),
                'frame_path': str(frame_path),
                'category': clip.get('category', ''),
                'caption': clip.get('caption', '')
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_video = Path(tmpdir) / "video.mp4"

            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                duration = end - start

                # Download video segment
                result = subprocess.run([
                    "yt-dlp",
                    "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
                    "--download-sections", f"*{start}-{end}",
                    "--force-keyframes-at-cuts",
                    "-o", str(tmp_video),
                    "--quiet",
                    "--no-warnings",
                    url
                ], capture_output=True, text=True, timeout=120)

                if not tmp_video.exists():
                    # Try simpler format
                    result = subprocess.run([
                        "yt-dlp",
                        "-f", "best",
                        "--download-sections", f"*{start}-{end}",
                        "--force-keyframes-at-cuts",
                        "-o", str(tmp_video),
                        "--quiet",
                        "--no-warnings",
                        url
                    ], capture_output=True, text=True, timeout=120)

                if not tmp_video.exists():
                    return None

                # Extract audio at 16kHz mono
                subprocess.run([
                    "ffmpeg",
                    "-i", str(tmp_video),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(self.config.audio_sample_rate),
                    "-ac", "1",
                    "-y",
                    str(audio_path)
                ], capture_output=True, check=True, timeout=60)

                # Extract middle frame at 256x256
                middle_sec = duration / 2
                subprocess.run([
                    "ffmpeg",
                    "-i", str(tmp_video),
                    "-ss", str(middle_sec),
                    "-vframes", "1",
                    "-vf", f"scale={self.config.frame_size}:{self.config.frame_size}:force_original_aspect_ratio=increase,crop={self.config.frame_size}:{self.config.frame_size}",
                    "-y",
                    str(frame_path)
                ], capture_output=True, check=True, timeout=60)

                if audio_path.exists() and frame_path.exists():
                    return {
                        'audio_path': str(audio_path),
                        'frame_path': str(frame_path),
                        'category': clip.get('category', ''),
                        'caption': clip.get('caption', '')
                    }

            except subprocess.TimeoutExpired:
                pass
            except subprocess.CalledProcessError:
                pass
            except Exception:
                pass

        # Clean up partial files
        if audio_path.exists():
            audio_path.unlink()
        if frame_path.exists():
            frame_path.unlink()

        return None

    def download_all(self, resume: bool = True) -> Dict[str, int]:
        """Download all filtered clips"""

        # Load or create filtered clips list
        if self.filtered_clips_path.exists() and resume:
            print(f"Loading filtered clips from {self.filtered_clips_path}...")
            with open(self.filtered_clips_path, 'r') as f:
                clips = json.load(f)
        else:
            print("Filtering clips from metadata...")
            clips = self.load_and_filter_clips()

            # Save filtered list for resume
            with open(self.filtered_clips_path, 'w') as f:
                json.dump(clips, f, indent=2)
            print(f"Saved filtered clips to {self.filtered_clips_path}")

        # Load existing progress
        log = self._load_progress() if resume else {}

        stats = {'downloaded': 0, 'failed': 0, 'skipped': 0}

        print(f"\nDownloading {len(clips)} clips...")
        for i, clip in enumerate(tqdm(clips, desc="Downloading")):
            clip_id = self._get_clip_id(clip)

            # Skip if already processed
            if clip_id in log:
                if log[clip_id].get('status') == 'success':
                    stats['skipped'] += 1
                    continue

            # Download
            result = self.download_clip(clip)

            if result:
                log[clip_id] = {
                    'status': 'success',
                    'video_id': clip['video_id'],
                    'start': clip['start'],
                    'end': clip['end'],
                    'category': result.get('category', ''),
                    'caption': result.get('caption', ''),
                    'audio_path': result['audio_path'],
                    'frame_path': result['frame_path']
                }
                stats['downloaded'] += 1
            else:
                log[clip_id] = {
                    'status': 'failed',
                    'video_id': clip['video_id'],
                    'start': clip['start'],
                    'end': clip['end']
                }
                stats['failed'] += 1

            # Checkpoint every 100 clips
            if (i + 1) % 100 == 0:
                self._save_progress(log)
                tqdm.write(f"  Checkpoint: {stats['downloaded']} downloaded, {stats['failed']} failed")

        # Final save
        self._save_progress(log)

        print(f"\nDownload complete:")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped (already done): {stats['skipped']}")

        return stats

    def get_successful_clips(self) -> List[Dict]:
        """Get list of successfully downloaded clips"""
        log = self._load_progress()
        return [
            entry for entry in log.values()
            if entry.get('status') == 'success'
        ]


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Download HowTo100M dataset subset")
    parser.add_argument("--data-dir", default="./data/howto100m", help="Data directory")
    parser.add_argument("--metadata-dir", default="./data/howto100m/metadata", help="Metadata directory")
    parser.add_argument("--target-clips", type=int, default=10000, help="Target clip count")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--check-metadata", action="store_true", help="Just check metadata files")
    args = parser.parse_args()

    config = HowTo100MConfig(
        data_dir=args.data_dir,
        metadata_dir=args.metadata_dir,
        target_clips=args.target_clips
    )

    downloader = HowTo100MDownloader(config)

    if args.check_metadata:
        files = downloader.find_metadata_files()
        print("Found metadata files:")
        for name, path in files.items():
            print(f"  {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return

    stats = downloader.download_all(resume=not args.no_resume)

    # Print summary
    successful = downloader.get_successful_clips()
    categories = set(c.get('category', 'unknown') for c in successful)
    print(f"\nReady for training: {len(successful)} clips across {len(categories)} categories")


if __name__ == "__main__":
    main()
