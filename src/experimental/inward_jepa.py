import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class InwardJEPA(nn.Module):
    """
    Learns memory-to-memory association structure.
    
    The predictor transforms an embedding into "association space" —
    embeddings that should associate will have similar predictions.
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Predictor network — this IS the associative map
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Transform embedding into association query space"""
        return self.predictor(embedding)
    
    def association_scores(self, 
                           query_embedding: torch.Tensor,
                           memory_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Score how strongly a query associates with each memory.
        
        Args:
            query_embedding: [D] or [1, D] — current perception
            memory_embeddings: [N, D] — all stored episode embeddings
            
        Returns:
            [N] association scores (higher = stronger association)
        """
        # Transform query through predictor
        query_pred = self.predictor(query_embedding)  # [D] or [1, D]
        
        if query_pred.dim() == 1:
            query_pred = query_pred.unsqueeze(0)
        
        # Cosine similarity to all memories
        # Memory embeddings stay fixed — we're not transforming them
        scores = F.cosine_similarity(
            query_pred.expand(memory_embeddings.size(0), -1),
            memory_embeddings,
            dim=1
        )
        return scores
    
    def get_warm_set(self,
                     query_embedding: torch.Tensor,
                     memory_embeddings: torch.Tensor,
                     top_k: int = 10,
                     threshold: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top associated memories (the "warm set").
        
        Returns:
            indices: [K] indices into memory_embeddings
            scores: [K] association scores
        """
        scores = self.association_scores(query_embedding, memory_embeddings)
        
        # Filter by threshold, then take top-k
        mask = scores > threshold
        valid_scores = scores.clone()
        valid_scores[~mask] = -float('inf')
        
        k = min(top_k, mask.sum().item())
        if k == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([])
        
        top_scores, top_indices = torch.topk(valid_scores, k)
        return top_indices, top_scores


class InwardJEPATrainer:
    """
    Trains the InwardJEPA using co-occurrence as supervision.
    
    Positive pairs: episodes sharing objects
    Negative pairs: episodes with no object overlap
    """
    
    def __init__(self, 
                 model: InwardJEPA,
                 learning_rate: float = 1e-4,
                 temperature: float = 0.07):
        self.model = model
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def contrastive_loss(self,
                         anchor: torch.Tensor,      # [B, D]
                         positive: torch.Tensor,    # [B, D]
                         negatives: torch.Tensor    # [B, N, D]
                        ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.
        
        Anchor transformed through predictor should be close to positive,
        far from negatives.
        """
        batch_size = anchor.size(0)
        
        # Transform anchors
        anchor_pred = self.model(anchor)  # [B, D]
        
        # Positive similarity
        pos_sim = F.cosine_similarity(anchor_pred, positive, dim=1)  # [B]
        pos_sim = pos_sim / self.temperature
        
        # Negative similarities
        anchor_pred_exp = anchor_pred.unsqueeze(1)  # [B, 1, D]
        neg_sim = F.cosine_similarity(anchor_pred_exp, negatives, dim=2)  # [B, N]
        neg_sim = neg_sim / self.temperature
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def train_step(self,
                   anchors: torch.Tensor,
                   positives: torch.Tensor,
                   negatives: torch.Tensor) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.contrastive_loss(anchors, positives, negatives)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def train_inward_jepa(episode_memory, model, trainer, epochs=10, batch_size=32):
    """Run during dreaming phase"""
    for epoch in range(epochs):
        batch = episode_memory.sample_training_batch(batch_size=batch_size)
        if batch is None:
            print(f"Not enough data for training (need more diverse episodes)")
            return
        
        anchors, positives, negatives = batch
        
        # Convert to tensors and move to model's device
        device = next(model.parameters()).device
        anchors_t = torch.from_numpy(anchors).float().to(device)
        positives_t = torch.from_numpy(positives).float().to(device)
        negatives_t = torch.from_numpy(negatives).float().to(device)
        
        loss = trainer.train_step(anchors_t, positives_t, negatives_t)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
