"""
ImageBind Distillation Evaluation

Tests the alignment between distilled student projections and ImageBind's
semantic space. Verifies that projected BEATs and V-JEPA embeddings can
match ImageBind text embeddings for retrieval tasks.

Key tests:
1. Text-to-Audio retrieval via projected BEATs
2. Text-to-Visual retrieval via projected V-JEPA
3. Cross-modal alignment (audio-visual pairs should be close)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class EvalResult:
    """Result from an evaluation test"""
    test_name: str
    metrics: Dict[str, float]
    details: Dict


class DistillationEvaluator:
    """
    Evaluates the quality of ImageBind distillation.

    Tests whether projected BEATs and V-JEPA embeddings land in
    ImageBind's semantic space with proper alignment.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None
    ):
        """
        Initialize evaluator with trained projection heads.

        Args:
            checkpoint_path: Path to distillation checkpoint
            device: Device to use for computation
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load projection heads
        from imagebind_distillation import load_projection_heads
        self.audio_proj, self.visual_proj = load_projection_heads(
            checkpoint_path, self.device
        )

        # Load ImageBind (for text encoding)
        from imagebind_encoder import ImageBindEncoder
        self.imagebind = ImageBindEncoder(self.device)

        # Lazy-load student encoders (only when needed)
        self._beats_encoder = None
        self._vjepa_encoder = None

    @property
    def beats_encoder(self):
        """Lazy-load BEATs encoder"""
        if self._beats_encoder is None:
            from beats_encoder import BEATsEncoder
            self._beats_encoder = BEATsEncoder()
        return self._beats_encoder

    @property
    def vjepa_encoder(self):
        """Lazy-load V-JEPA encoder"""
        if self._vjepa_encoder is None:
            from vjepa_encoder import VJEPAEncoder
            self._vjepa_encoder = VJEPAEncoder()
        return self._vjepa_encoder

    @torch.no_grad()
    def encode_text_imagebind(self, texts: List[str]) -> torch.Tensor:
        """Encode text using ImageBind"""
        return self.imagebind.encode_text(texts).to(self.device)

    @torch.no_grad()
    def encode_audio_projected(self, audio_path: str) -> torch.Tensor:
        """Encode audio using BEATs + projection to ImageBind space"""
        beats_emb = self.beats_encoder.encode_file(audio_path)
        beats_emb = beats_emb.to(self.device)
        if beats_emb.dim() == 1:
            beats_emb = beats_emb.unsqueeze(0)
        return self.audio_proj(beats_emb)

    @torch.no_grad()
    def encode_vision_projected(self, image_path: str) -> torch.Tensor:
        """Encode image using V-JEPA + projection to ImageBind space"""
        import cv2
        import numpy as np

        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames = np.stack([frame] * 16)  # V-JEPA needs video

        vjepa_emb = self.vjepa_encoder.encode_frames(frames)
        vjepa_emb = vjepa_emb.to(self.device)
        if vjepa_emb.dim() == 1:
            vjepa_emb = vjepa_emb.unsqueeze(0)
        return self.visual_proj(vjepa_emb)

    @staticmethod
    def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings"""
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        return (emb1 @ emb2.T).squeeze()

    def test_text_audio_alignment(
        self,
        audio_path: str,
        ground_truth_text: str,
        distractor_texts: List[str]
    ) -> EvalResult:
        """
        Test if projected audio embedding aligns with correct text.

        Args:
            audio_path: Path to audio file
            ground_truth_text: Text description matching the audio
            distractor_texts: List of incorrect text descriptions

        Returns:
            EvalResult with retrieval metrics
        """
        # Encode audio
        audio_emb = self.encode_audio_projected(audio_path)

        # Encode all texts
        all_texts = [ground_truth_text] + distractor_texts
        text_embs = self.encode_text_imagebind(all_texts)

        # Compute similarities
        sims = self.cosine_similarity(audio_emb, text_embs)

        # Get ranking
        ranked_indices = torch.argsort(sims, descending=True).cpu().tolist()
        rank_of_gt = ranked_indices.index(0) + 1  # 1-indexed rank

        # Compute metrics
        metrics = {
            'rank_of_ground_truth': rank_of_gt,
            'gt_similarity': sims[0].item(),
            'mean_distractor_similarity': sims[1:].mean().item(),
            'is_top1': rank_of_gt == 1,
            'reciprocal_rank': 1.0 / rank_of_gt
        }

        details = {
            'audio_path': audio_path,
            'ground_truth_text': ground_truth_text,
            'distractor_texts': distractor_texts,
            'all_similarities': sims.cpu().tolist(),
            'ranking': ranked_indices
        }

        return EvalResult(
            test_name='text_audio_alignment',
            metrics=metrics,
            details=details
        )

    def test_text_visual_alignment(
        self,
        image_path: str,
        ground_truth_text: str,
        distractor_texts: List[str]
    ) -> EvalResult:
        """
        Test if projected visual embedding aligns with correct text.

        Args:
            image_path: Path to image file
            ground_truth_text: Text description matching the image
            distractor_texts: List of incorrect text descriptions

        Returns:
            EvalResult with retrieval metrics
        """
        # Encode image
        visual_emb = self.encode_vision_projected(image_path)

        # Encode all texts
        all_texts = [ground_truth_text] + distractor_texts
        text_embs = self.encode_text_imagebind(all_texts)

        # Compute similarities
        sims = self.cosine_similarity(visual_emb, text_embs)

        # Get ranking
        ranked_indices = torch.argsort(sims, descending=True).cpu().tolist()
        rank_of_gt = ranked_indices.index(0) + 1

        metrics = {
            'rank_of_ground_truth': rank_of_gt,
            'gt_similarity': sims[0].item(),
            'mean_distractor_similarity': sims[1:].mean().item(),
            'is_top1': rank_of_gt == 1,
            'reciprocal_rank': 1.0 / rank_of_gt
        }

        details = {
            'image_path': image_path,
            'ground_truth_text': ground_truth_text,
            'distractor_texts': distractor_texts,
            'all_similarities': sims.cpu().tolist(),
            'ranking': ranked_indices
        }

        return EvalResult(
            test_name='text_visual_alignment',
            metrics=metrics,
            details=details
        )

    def test_audio_visual_alignment(
        self,
        audio_path: str,
        matching_image_path: str,
        non_matching_image_paths: List[str]
    ) -> EvalResult:
        """
        Test if projected audio and visual embeddings from same source
        are closer than non-matching pairs.

        Args:
            audio_path: Path to audio file
            matching_image_path: Path to image from same clip
            non_matching_image_paths: Paths to images from different clips

        Returns:
            EvalResult with alignment metrics
        """
        # Encode audio
        audio_emb = self.encode_audio_projected(audio_path)

        # Encode all images
        all_image_paths = [matching_image_path] + non_matching_image_paths
        visual_embs = []
        for img_path in all_image_paths:
            visual_embs.append(self.encode_vision_projected(img_path))
        visual_embs = torch.cat(visual_embs, dim=0)

        # Compute similarities
        sims = self.cosine_similarity(audio_emb, visual_embs)

        # Get ranking
        ranked_indices = torch.argsort(sims, descending=True).cpu().tolist()
        rank_of_match = ranked_indices.index(0) + 1

        metrics = {
            'rank_of_matching': rank_of_match,
            'matching_similarity': sims[0].item(),
            'mean_nonmatching_similarity': sims[1:].mean().item(),
            'is_top1': rank_of_match == 1,
            'reciprocal_rank': 1.0 / rank_of_match
        }

        details = {
            'audio_path': audio_path,
            'matching_image_path': matching_image_path,
            'non_matching_image_paths': non_matching_image_paths,
            'all_similarities': sims.cpu().tolist(),
            'ranking': ranked_indices
        }

        return EvalResult(
            test_name='audio_visual_alignment',
            metrics=metrics,
            details=details
        )

    def run_concept_test(
        self,
        concept: str,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        distractors: Optional[List[str]] = None
    ) -> Dict[str, EvalResult]:
        """
        Run a comprehensive concept alignment test.

        Tests that text, audio, and visual representations of the same
        concept are aligned in the shared embedding space.

        Args:
            concept: Text description of the concept (e.g., "hammer hitting nail")
            audio_path: Optional path to audio of the concept
            image_path: Optional path to image of the concept
            distractors: Optional distractor texts (default: general concepts)

        Returns:
            Dictionary of test name -> EvalResult
        """
        if distractors is None:
            distractors = [
                "a dog barking",
                "water flowing",
                "music playing",
                "typing on keyboard",
                "birds chirping"
            ]

        results = {}

        # Get text embedding via ImageBind
        text_emb = self.encode_text_imagebind([concept])

        if audio_path:
            # Test audio-to-text alignment
            audio_emb = self.encode_audio_projected(audio_path)
            sim = self.cosine_similarity(audio_emb, text_emb).item()

            results['audio_text_similarity'] = EvalResult(
                test_name='audio_text_similarity',
                metrics={
                    'cosine_similarity': sim,
                    'concept': concept
                },
                details={'audio_path': audio_path}
            )

            # Run full retrieval test
            results['audio_retrieval'] = self.test_text_audio_alignment(
                audio_path, concept, distractors
            )

        if image_path:
            # Test visual-to-text alignment
            visual_emb = self.encode_vision_projected(image_path)
            sim = self.cosine_similarity(visual_emb, text_emb).item()

            results['visual_text_similarity'] = EvalResult(
                test_name='visual_text_similarity',
                metrics={
                    'cosine_similarity': sim,
                    'concept': concept
                },
                details={'image_path': image_path}
            )

            # Run full retrieval test
            results['visual_retrieval'] = self.test_text_visual_alignment(
                image_path, concept, distractors
            )

        if audio_path and image_path:
            # Test audio-visual alignment
            audio_emb = self.encode_audio_projected(audio_path)
            visual_emb = self.encode_vision_projected(image_path)
            sim = self.cosine_similarity(audio_emb, visual_emb).item()

            results['audio_visual_similarity'] = EvalResult(
                test_name='audio_visual_similarity',
                metrics={
                    'cosine_similarity': sim,
                    'concept': concept
                },
                details={
                    'audio_path': audio_path,
                    'image_path': image_path
                }
            )

        return results


def run_evaluation(
    checkpoint_path: str,
    data_dir: str = "./data",
    output_path: Optional[str] = None,
    device: str = None
):
    """
    Run full evaluation suite on distillation checkpoint.

    Args:
        checkpoint_path: Path to trained distillation checkpoint
        data_dir: Directory containing VGGSound/HowTo100M data
        output_path: Optional path to save results JSON
        device: Device to use
    """
    print("\n" + "=" * 60)
    print("ImageBind Distillation Evaluation")
    print("=" * 60)

    # Load evaluator
    print(f"\nLoading checkpoint: {checkpoint_path}")
    evaluator = DistillationEvaluator(checkpoint_path, device)

    # Find test samples
    data_dir = Path(data_dir)
    test_samples = []

    # Check VGGSound
    vggsound_log = data_dir / "vggsound" / "download_log.json"
    if vggsound_log.exists():
        with open(vggsound_log) as f:
            log = json.load(f)
        for clip_id, entry in log.items():
            if entry.get('status') == 'success':
                # Use 20% split for testing
                if hash(clip_id) % 100 >= 80:
                    test_samples.append({
                        'source': 'vggsound',
                        'clip_id': clip_id,
                        'audio_path': entry['audio_path'],
                        'frame_path': entry['frame_path'],
                        'label': entry['label']
                    })

    # Check HowTo100M
    howto_log = data_dir / "howto100m" / "download_log.json"
    if howto_log.exists():
        with open(howto_log) as f:
            log = json.load(f)
        for clip_id, entry in log.items():
            if entry.get('status') == 'success':
                if hash(clip_id) % 100 >= 80:
                    test_samples.append({
                        'source': 'howto100m',
                        'clip_id': clip_id,
                        'audio_path': entry['audio_path'],
                        'frame_path': entry['frame_path'],
                        'label': entry.get('category', 'unknown')
                    })

    if not test_samples:
        print("No test samples found. Running synthetic test...")
        run_synthetic_test(evaluator)
        return

    print(f"\nFound {len(test_samples)} test samples")

    # Run evaluation
    all_results = {
        'audio_retrieval': [],
        'visual_retrieval': [],
        'audio_visual_alignment': []
    }

    # Get unique labels for distractors
    all_labels = list(set(s['label'] for s in test_samples))

    for i, sample in enumerate(test_samples[:50]):  # Limit to 50 samples
        print(f"\rEvaluating sample {i+1}/50...", end="", flush=True)

        # Get distractors (other labels)
        distractors = [l for l in all_labels if l != sample['label']][:5]

        if len(distractors) < 5:
            distractors.extend([
                "a dog barking",
                "water flowing",
                "music playing"
            ][:5 - len(distractors)])

        try:
            # Audio-to-text retrieval
            result = evaluator.test_text_audio_alignment(
                sample['audio_path'],
                sample['label'],
                distractors
            )
            all_results['audio_retrieval'].append(result.metrics)

            # Visual-to-text retrieval
            result = evaluator.test_text_visual_alignment(
                sample['frame_path'],
                sample['label'],
                distractors
            )
            all_results['visual_retrieval'].append(result.metrics)

        except Exception as e:
            print(f"\nError processing {sample['clip_id']}: {e}")

    # Compute aggregate metrics
    print("\n\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for test_name, results in all_results.items():
        if not results:
            continue

        print(f"\n{test_name.upper()}:")
        print(f"  Samples evaluated: {len(results)}")

        # Compute means
        mean_rank = np.mean([r['rank_of_ground_truth'] for r in results])
        top1_acc = np.mean([r['is_top1'] for r in results])
        mrr = np.mean([r['reciprocal_rank'] for r in results])
        mean_gt_sim = np.mean([r['gt_similarity'] for r in results])

        print(f"  Mean Rank: {mean_rank:.2f}")
        print(f"  Top-1 Accuracy: {top1_acc:.2%}")
        print(f"  Mean Reciprocal Rank: {mrr:.4f}")
        print(f"  Mean Ground Truth Similarity: {mean_gt_sim:.4f}")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'checkpoint_path': checkpoint_path,
                'num_samples': len(test_samples),
                'results': {
                    name: [r for r in results]
                    for name, results in all_results.items()
                }
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def run_synthetic_test(evaluator: DistillationEvaluator):
    """
    Run synthetic evaluation when no data is available.

    Tests basic functionality by checking if projections produce
    reasonable embeddings (correct shape, normalized, etc.)
    """
    print("\n" + "=" * 60)
    print("Synthetic Evaluation (No Data)")
    print("=" * 60)

    # Test projection shapes
    print("\n1. Testing projection layer functionality...")

    # Create synthetic inputs
    dummy_beats = torch.randn(4, 768).to(evaluator.device)
    dummy_vjepa = torch.randn(4, 1024).to(evaluator.device)

    # Project
    proj_audio = evaluator.audio_proj(dummy_beats)
    proj_visual = evaluator.visual_proj(dummy_vjepa)

    print(f"  BEATs input: {dummy_beats.shape} -> output: {proj_audio.shape}")
    print(f"  V-JEPA input: {dummy_vjepa.shape} -> output: {proj_visual.shape}")

    # Check normalization
    audio_norms = proj_audio.norm(dim=-1)
    visual_norms = proj_visual.norm(dim=-1)

    print(f"  Audio embedding norms: {audio_norms.mean():.4f} +/- {audio_norms.std():.4f}")
    print(f"  Visual embedding norms: {visual_norms.mean():.4f} +/- {visual_norms.std():.4f}")

    # Test ImageBind text encoding
    print("\n2. Testing ImageBind text encoding...")
    test_texts = ["a hammer", "a cat", "music playing", "water flowing"]
    text_embs = evaluator.encode_text_imagebind(test_texts)
    print(f"  Text embeddings shape: {text_embs.shape}")

    # Compute text-text similarities
    print("\n3. Text embedding sanity check:")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = evaluator.cosine_similarity(text_embs[i], text_embs[j])
            print(f"  '{test_texts[i]}' <-> '{test_texts[j]}': {sim.item():.3f}")

    print("\n" + "=" * 60)
    print("Synthetic test complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ImageBind distillation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/imagebind_distillation/best_checkpoint.pt",
        help="Path to distillation checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing VGGSound/HowTo100M data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        device=args.device
    )
