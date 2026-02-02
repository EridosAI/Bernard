"""
V-JEPA Encoder for Distillation

Standalone V-JEPA encoder that doesn't import unnecessary dependencies.
Used for ImageBind distillation training.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModel, AutoVideoProcessor
from peft import PeftModel


class VJEPAEncoder:
    """
    V-JEPA encoder for visual embeddings.

    Produces 1024-dimensional embeddings from video frames.
    Can optionally use a LoRA adapter for fine-tuned representations.
    """

    def __init__(
        self,
        model_path: str = "./models/base/vjepa2",
        adapter_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize V-JEPA encoder.

        Args:
            model_path: Path to pretrained V-JEPA model
            adapter_path: Optional path to LoRA adapter
            device: Device to use (defaults to CUDA if available)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading V-JEPA from {model_path}...")

        # Load base model
        base_model = AutoModel.from_pretrained(model_path)

        # Apply adapter if provided
        if adapter_path and Path(adapter_path).exists():
            print(f"  Applying LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
            if adapter_path:
                print(f"  Note: Adapter path {adapter_path} not found, using base model")

        self.model = self.model.to(device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(model_path)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        print("  V-JEPA loaded and frozen")

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Encode video frames to embedding.

        Args:
            frames: Video frames of shape [T, H, W, 3] (RGB, uint8 or float)

        Returns:
            Embedding tensor of shape [1024] on CPU
        """
        # Process video
        inputs = self.processor(videos=list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        output = self.model(**inputs)

        # Mean pool over sequence
        embedding = output.last_hidden_state.mean(dim=1)

        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def encode_region(
        self,
        frames: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 20
    ) -> torch.Tensor:
        """
        Encode a specific region across frames.

        Args:
            frames: Video frames of shape [T, H, W, 3]
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Pixels to pad around bbox

        Returns:
            Embedding tensor of shape [1024] on CPU
        """
        x1, y1, x2, y2 = bbox
        h, w = frames.shape[1:3]

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop and resize
        cropped = frames[:, y1:y2, x1:x2, :]
        resized = np.array([cv2.resize(f, (256, 256)) for f in cropped])

        return self.encode_frames(resized)

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode a single image by repeating it as a video.

        Args:
            image_path: Path to image file

        Returns:
            Embedding tensor of shape [1024] on CPU
        """
        # Load image
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # V-JEPA expects video, so repeat frame 16 times
        frames = np.stack([frame] * 16)

        return self.encode_frames(frames)


def test_vjepa():
    """Test V-JEPA encoder"""
    print("\n" + "=" * 60)
    print("V-JEPA Encoder Test")
    print("=" * 60)

    # Find adapter if exists
    adapters_dir = Path("./models/adapters")
    adapter_path = None
    if adapters_dir.exists():
        adapters = list(adapters_dir.glob("workshop_lora_*"))
        if adapters:
            adapter_path = str(sorted(adapters)[-1])  # Most recent
            print(f"Found adapter: {adapter_path}")

    # Initialize encoder
    encoder = VJEPAEncoder(adapter_path=adapter_path)

    # Test with synthetic frames
    print("\nTesting with synthetic frames...")
    frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)
    embedding = encoder.encode_frames(frames)
    print(f"  Output shape: {embedding.shape}")
    print(f"  Output norm: {embedding.norm().item():.4f}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_vjepa()
