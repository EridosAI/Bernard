#!/usr/bin/env python3
"""
Train Audio-Visual Projection on Combined HowTo100M + VGGSound Dataset

This script trains a projection head to align BEATs audio embeddings with
V-JEPA visual embeddings using both VGGSound and HowTo100M datasets.

Configuration:
    - 80/20 train/test split (deterministic)
    - Batch size: 32
    - Learning rate: 1e-4, AdamW, weight_decay=0.01
    - Temperature: 0.07
    - Early stopping: patience=10 on validation Top-1

Usage:
    python scripts/train_combined_av.py
    python scripts/train_combined_av.py --eval-only
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

import argparse
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Train audio-visual projection on combined HowTo100M + VGGSound"
    )
    parser.add_argument(
        "--vggsound-dir",
        default="./data/vggsound",
        help="VGGSound data directory"
    )
    parser.add_argument(
        "--howto100m-dir",
        default="./data/howto100m",
        help="HowTo100M data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./models/adapters/audio_visual_projection_combined",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs (default: 100, early stopping may trigger earlier)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Evaluate every N epochs (default: 5)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint name for eval-only mode (default: best)"
    )
    args = parser.parse_args()

    # Change to project directory
    os.chdir(project_dir)

    print("=" * 70)
    print("Audio-Visual Training on Combined Dataset")
    print("=" * 70)
    print(f"VGGSound dir: {args.vggsound_dir}")
    print(f"HowTo100M dir: {args.howto100m_dir}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Import training modules
    from scripts.audio_visual_training import AudioVisualTrainingConfig, AudioVisualTrainer

    # Create config for combined training
    config = AudioVisualTrainingConfig(
        # Use combined datasets
        use_combined=True,
        vggsound_dir=args.vggsound_dir,
        howto100m_dir=args.howto100m_dir,

        # Training params
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        temperature=0.07,

        # Early stopping
        early_stopping_patience=args.patience,
        eval_every=args.eval_every,

        # Output
        output_dir=args.output_dir
    )

    # Create trainer
    trainer = AudioVisualTrainer(config)

    if args.eval_only:
        # Evaluation-only mode
        print("Running evaluation only...")
        trainer.setup()
        trainer.load_checkpoint(args.checkpoint)
        metrics = trainer.evaluate_retrieval()

        print("\n" + "=" * 70)
        print("Retrieval Metrics (Combined Dataset)")
        print("=" * 70)
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.3f} ({metrics['top1_accuracy']*100:.1f}%)")
        print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.3f} ({metrics['top5_accuracy']*100:.1f}%)")
        print(f"  Mean Rank: {metrics['mean_rank']:.1f}")

        # Compare to baseline
        baseline_top1 = 0.196
        improvement = metrics['top1_accuracy'] - baseline_top1
        print(f"\nBaseline (VGGSound-only): {baseline_top1*100:.1f}% Top-1")
        print(f"Improvement: {improvement*100:+.1f}%")
    else:
        # Training mode
        print("Starting combined training...")
        history = trainer.train()

        # Print final comparison
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)

        best_top1 = history['best_top1_accuracy']
        baseline_top1 = 0.196

        print(f"VGGSound-only baseline: {baseline_top1*100:.1f}% Top-1")
        print(f"Combined dataset result: {best_top1*100:.1f}% Top-1")
        print(f"Improvement: {(best_top1 - baseline_top1)*100:+.1f}%")

        # Save comparison summary
        summary = {
            'baseline_vggsound_only': {
                'top1_accuracy': baseline_top1,
                'description': 'VGGSound-only training (270 clips)'
            },
            'combined_training': {
                'top1_accuracy': best_top1,
                'best_epoch': history['best_epoch'],
                'early_stopped': history.get('early_stopped', False),
                'description': 'Combined HowTo100M + VGGSound (2282 clips)'
            },
            'improvement': best_top1 - baseline_top1,
            'timestamp': datetime.now().isoformat()
        }

        summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
