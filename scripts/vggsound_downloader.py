"""
VGGSound Dataset Downloader

Downloads a subset of VGGSound for audio-visual contrastive learning.
Uses yt-dlp to download YouTube clips, extracts audio at 16kHz and representative frames.
"""

import os
import json
import subprocess
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import csv
import tempfile
import shutil


@dataclass
class DownloadConfig:
    """Configuration for VGGSound download"""
    output_dir: str = "./data/vggsound"
    clips_per_class: int = 50
    audio_sample_rate: int = 16000
    frame_size: int = 256
    clip_duration: int = 10  # seconds

    # 20 diverse audio categories
    target_classes: List[str] = field(default_factory=lambda: [
        "playing acoustic guitar",
        "dog barking",
        "car engine starting",
        "typing on computer keyboard",
        "door knock",
        "hammering",
        "drilling",
        "sawing wood",
        "glass breaking",
        "water running from a tap",
        "cat meowing",
        "bird chirping",
        "telephone ringing",
        "clock ticking",
        "thunder",
        "rain on surface",
        "baby crying",
        "people laughing",
        "people coughing",
        "people clapping",
    ])


class VGGSoundDownloader:
    """Downloads VGGSound audio-visual pairs"""

    VGGSOUND_CSV_URL = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"

    def __init__(self, config: DownloadConfig = None):
        self.config = config or DownloadConfig()
        self.output_dir = Path(self.config.output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.frames_dir = self.output_dir / "frames"
        self.log_path = self.output_dir / "download_log.json"
        self.csv_path = self.output_dir / "vggsound.csv"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

    def download_csv(self) -> bool:
        """Download VGGSound CSV if not present"""
        if self.csv_path.exists():
            print(f"CSV already exists: {self.csv_path}")
            return True

        print("Downloading VGGSound CSV...")
        try:
            result = subprocess.run(
                ["curl", "-L", "-o", str(self.csv_path), self.VGGSOUND_CSV_URL],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Downloaded: {self.csv_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to download CSV: {e.stderr}")
            return False

    def load_csv(self) -> List[Dict]:
        """Load and parse VGGSound CSV"""
        entries = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    entries.append({
                        'video_id': row[0],
                        'start_sec': int(row[1]),
                        'label': row[2]
                    })
        return entries

    def select_clips(self) -> List[Dict]:
        """Select balanced subset of clips from target classes"""
        all_entries = self.load_csv()

        # Group by label
        by_label = {}
        for entry in all_entries:
            label = entry['label']
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(entry)

        # Find matching classes (fuzzy match)
        selected = []
        for target in self.config.target_classes:
            # Try exact match first
            if target in by_label:
                pool = by_label[target]
            else:
                # Fuzzy match - find labels containing target words
                matching_labels = [l for l in by_label.keys()
                                   if target.lower() in l.lower() or
                                   all(word in l.lower() for word in target.lower().split())]
                if matching_labels:
                    pool = by_label[matching_labels[0]]
                    print(f"  '{target}' -> '{matching_labels[0]}' ({len(pool)} clips)")
                else:
                    print(f"  Warning: No match for '{target}'")
                    continue

            # Random sample
            sample_size = min(self.config.clips_per_class, len(pool))
            selected.extend(random.sample(pool, sample_size))

        print(f"\nSelected {len(selected)} clips from {len(self.config.target_classes)} classes")
        return selected

    def _load_progress(self) -> Dict:
        """Load download progress from log"""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_progress(self, log: Dict):
        """Save download progress"""
        with open(self.log_path, 'w') as f:
            json.dump(log, f, indent=2)

    def _get_clip_id(self, entry: Dict) -> str:
        """Generate unique clip ID"""
        return f"{entry['video_id']}_{entry['start_sec']}"

    def download_clip(self, entry: Dict) -> Optional[Dict]:
        """
        Download single clip, extract audio and frame.

        Returns dict with paths on success, None on failure.
        """
        video_id = entry['video_id']
        start_sec = entry['start_sec']
        label = entry['label']
        clip_id = self._get_clip_id(entry)

        audio_path = self.audio_dir / f"{clip_id}.wav"
        frame_path = self.frames_dir / f"{clip_id}.jpg"

        # Skip if already exists
        if audio_path.exists() and frame_path.exists():
            return {
                'audio_path': str(audio_path),
                'frame_path': str(frame_path),
                'label': label
            }

        # Create temp directory for download
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_video = Path(tmpdir) / "video.mp4"

            try:
                # Download video segment with yt-dlp
                # Use --download-sections to get only the needed portion
                url = f"https://www.youtube.com/watch?v={video_id}"
                end_sec = start_sec + self.config.clip_duration

                result = subprocess.run([
                    "yt-dlp",
                    "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
                    "--download-sections", f"*{start_sec}-{end_sec}",
                    "--force-keyframes-at-cuts",
                    "-o", str(tmp_video),
                    "--quiet",
                    "--no-warnings",
                    url
                ], capture_output=True, text=True, timeout=120)

                if not tmp_video.exists():
                    # Try alternative format
                    result = subprocess.run([
                        "yt-dlp",
                        "-f", "best",
                        "--download-sections", f"*{start_sec}-{end_sec}",
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
                # Get video duration and extract frame at midpoint
                middle_sec = self.config.clip_duration / 2
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
                        'label': label
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
        """
        Download all selected clips.

        Returns stats dict: {downloaded, failed, skipped}
        """
        # Download CSV if needed
        if not self.download_csv():
            return {'downloaded': 0, 'failed': 0, 'skipped': 0}

        # Select clips
        print("\nSelecting clips...")
        clips = self.select_clips()

        # Load existing progress
        log = self._load_progress() if resume else {}

        stats = {'downloaded': 0, 'failed': 0, 'skipped': 0}

        print(f"\nDownloading {len(clips)} clips...")
        for i, entry in enumerate(tqdm(clips, desc="Downloading")):
            clip_id = self._get_clip_id(entry)

            # Skip if already processed
            if clip_id in log:
                if log[clip_id].get('status') == 'success':
                    stats['skipped'] += 1
                    continue

            # Download
            result = self.download_clip(entry)

            if result:
                log[clip_id] = {
                    'status': 'success',
                    'video_id': entry['video_id'],
                    'start_sec': entry['start_sec'],
                    'label': entry['label'],
                    'audio_path': result['audio_path'],
                    'frame_path': result['frame_path']
                }
                stats['downloaded'] += 1
            else:
                log[clip_id] = {
                    'status': 'failed',
                    'video_id': entry['video_id'],
                    'start_sec': entry['start_sec'],
                    'label': entry['label']
                }
                stats['failed'] += 1

            # Checkpoint every 50 clips
            if (i + 1) % 50 == 0:
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
        """Get list of successfully downloaded clips from log"""
        log = self._load_progress()
        return [
            entry for entry in log.values()
            if entry.get('status') == 'success'
        ]


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Download VGGSound dataset subset")
    parser.add_argument("--output-dir", default="./data/vggsound", help="Output directory")
    parser.add_argument("--clips-per-class", type=int, default=50, help="Clips per class")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (don't resume)")
    args = parser.parse_args()

    config = DownloadConfig(
        output_dir=args.output_dir,
        clips_per_class=args.clips_per_class
    )

    downloader = VGGSoundDownloader(config)
    stats = downloader.download_all(resume=not args.no_resume)

    # Print summary
    successful = downloader.get_successful_clips()
    labels = set(c['label'] for c in successful)
    print(f"\nReady for training: {len(successful)} clips across {len(labels)} classes")


if __name__ == "__main__":
    main()
