"""
ALAN v4 — RLHF Reward Model
ECONX GROUP (PTY) LTD

Reward model for Reinforcement Learning from Human Feedback (RLHF).
Architecture: Same as ALAN but with a scalar output head.

Criteria the reward model learns to evaluate:
- Helpfulness
- Accuracy
- Engagement quality
- Topic tracking
- Emotional appropriateness
- Safety
- Creativity (when appropriate)
- Confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# Reward dimensions
REWARD_CRITERIA = [
    "helpfulness",
    "accuracy",
    "engagement_quality",
    "topic_tracking",
    "emotional_appropriateness",
    "safety",
    "creativity",
    "confidence_calibration",
]


class RewardModel(nn.Module):
    """
    Reward model for RLHF training.

    Takes a (prompt, response) pair and outputs a scalar reward score.
    Also provides per-dimension scores for interpretability.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int = 50257,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size + 100, hidden_dim)

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Scalar reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # Per-dimension reward heads (for interpretability)
        self.dimension_heads = nn.ModuleDict({
            criterion: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            for criterion in REWARD_CRITERIA
        })

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reward score for input sequence.

        Args:
            input_ids: (B, T) token IDs (prompt + response concatenated)
            attention_mask: (B, T) attention mask

        Returns:
            reward: (B, 1) scalar reward
            dimensions: dict of per-criterion scores
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Transformer encoding
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Pool: use last token representation
        pooled = x[:, -1, :]  # (B, hidden_dim)

        # Scalar reward
        reward = self.reward_head(pooled)  # (B, 1)

        # Per-dimension scores
        dimensions = {
            name: head(pooled)
            for name, head in self.dimension_heads.items()
        }

        return reward, dimensions


class PPOTrainer:
    """
    PPO trainer for RLHF alignment.

    Uses the reward model to train ALAN via Proximal Policy Optimization.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: RewardModel,
        ref_model: nn.Module,
        learning_rate: float = 1e-6,
        kl_penalty: float = 0.02,
        clip_range: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.kl_penalty = kl_penalty
        self.clip_range = clip_range
        self.device = device

        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        logger.info("[PPO] Trainer initialized")
        logger.info(f"  KL penalty: {kl_penalty}")
        logger.info(f"  Clip range: {clip_range}")

    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute rewards for generated responses."""
        # Concatenate prompt and response
        full_ids = torch.cat([input_ids, response_ids], dim=1).to(self.device)

        with torch.no_grad():
            reward, dimensions = self.reward_model(full_ids)

        return reward, {k: v.item() for k, v in dimensions.items()}

    def compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model."""
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        kl = F.kl_div(
            policy_probs.log(),
            ref_probs,
            reduction="batchmean",
            log_target=False,
        )
        return kl

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> Dict:
        """
        Single PPO training step.

        Args:
            prompt_ids: (B, T_prompt) prompt token IDs
            response_ids: (B, T_response) response token IDs

        Returns:
            training metrics
        """
        self.policy.train()

        # Get rewards
        rewards, dim_scores = self.compute_rewards(prompt_ids, response_ids)

        # Get policy logits
        full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(self.device)
        policy_logits, _ = self.policy(full_ids)

        # Get reference logits (frozen)
        with torch.no_grad():
            ref_logits, _ = self.ref_model(full_ids)

        # KL penalty
        kl = self.compute_kl_divergence(policy_logits, ref_logits)

        # PPO objective: reward - kl_penalty * kl
        loss = -(rewards.mean() - self.kl_penalty * kl)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "kl_divergence": kl.item(),
            "dimension_scores": dim_scores,
        }
