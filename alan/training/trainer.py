"""
ALAN v4 — Training Pipeline
ECONX GROUP (PTY) LTD

Complete training pipeline including:
- Dataset loading from JSONL/TXT training data
- Tokenization with ALAN's custom BPE tokenizer
- Curriculum-based training (4 phases)
- Image understanding training support
- CUDA/MPS/CPU device auto-detection
- Checkpointing and evaluation
- Loss logging and progress tracking

Usage:
    python training/trainer.py --size small --epochs 3
"""

import os
import sys
import json
import math
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.core_transformer import Alan, AlanConfig, build_alan, get_device, print_device_info
from config.guardrails import get_awareness_layer
from model.modules.emotional_intelligence import EMOTIONAL_TONES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training/training.log"),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# TOKENIZER
# ============================================================

def get_tokenizer():
    """Load ALAN's custom BPE tokenizer."""
    try:
        from model.tokenizer import get_tokenizer as _get_tokenizer
        tokenizer = _get_tokenizer()
        logger.info(f"[Tokenizer] ALAN tokenizer loaded: {tokenizer}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


# ============================================================
# DATASET
# ============================================================

class AlanDataset(Dataset):
    """
    Dataset for ALAN v4 training.
    Loads from JSONL training data files.
    Supports text and image+text examples.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_seq_len: int = 512,
        phase: str = "all",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.examples: List[Dict] = []

        data_path = Path(data_dir)
        self._load_data(data_path)
        logger.info(f"[Dataset] Loaded {len(self.examples)} examples (phase={phase})")

    def _load_data(self, data_path: Path):
        """Load all JSONL files from the data directory."""
        jsonl_files = list(data_path.glob("**/*.jsonl"))

        if not jsonl_files:
            logger.warning(f"No JSONL files found in {data_path}. Using synthetic data.")
            self._generate_synthetic_data()
            return

        for filepath in jsonl_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                example = json.loads(line)
                                record = self._build_record(example)
                                if record is not None:
                                    self.examples.append(record)
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")

    def _build_record(self, example: Dict) -> Optional[Dict]:
        text = self._extract_text(example)
        if not text or len(text) <= 10:
            return None

        record: Dict = {
            "text": text,
            "router_target": None,
            "tone_label": None,
            "engagement_target": None,
            "confidence_target": None,
            "should_ask_target": None,
            "dopamine_target": None,
            "practice_level": None,
            "knowledge_confidence_target": None,
        }

        ex_type = example.get("type", "")
        if ex_type in {"reasoning", "emotional_intelligence"}:
            module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]
            target = torch.zeros(len(module_names), dtype=torch.float32)
            if ex_type == "reasoning":
                target[module_names.index("reasoning")] = 1.0
                target[module_names.index("meta")] = 1.0
            if ex_type == "emotional_intelligence":
                target[module_names.index("emotional")] = 1.0
                target[module_names.index("meta")] = 0.5
            record["router_target"] = target

        if ex_type == "emotional_intelligence":
            tone = str(example.get("user_tone", "")).strip().lower()
            if tone in EMOTIONAL_TONES:
                record["tone_label"] = EMOTIONAL_TONES.index(tone)
            dims = example.get("emotional_dimensions", {})
            if isinstance(dims, dict):
                try:
                    empathy = float(dims.get("empathy_shown"))
                    directness = float(dims.get("directness"))
                    encouragement = float(dims.get("encouragement"))
                    record["engagement_target"] = torch.tensor(
                        [empathy, directness, encouragement],
                        dtype=torch.float32,
                    )
                except (TypeError, ValueError):
                    pass

        if ex_type == "epistemic":
            try:
                record["confidence_target"] = float(example.get("confidence_target"))
            except (TypeError, ValueError):
                record["confidence_target"] = None
            try:
                record["should_ask_target"] = float(example.get("should_ask_target"))
            except (TypeError, ValueError):
                record["should_ask_target"] = None

        # Dopamine target: expected reward signal (0-1) for this example
        try:
            dt = example.get("dopamine_target")
            if dt is not None:
                record["dopamine_target"] = float(dt)
        except (TypeError, ValueError):
            pass

        # Practice level: which rehearsal stage this example represents
        # Valid values: heard_once, restated, applied, connected, questioned, taught_back
        pl = example.get("practice_level")
        if isinstance(pl, str) and pl in {
            "heard_once", "restated", "applied",
            "connected", "questioned", "taught_back",
        }:
            from model.modules.practice_rehearsal import REHEARSAL_STAGES
            record["practice_level"] = REHEARSAL_STAGES[pl]

        # Knowledge confidence target: how confident the model SHOULD be (0-1)
        try:
            kct = example.get("knowledge_confidence_target")
            if kct is not None:
                record["knowledge_confidence_target"] = float(kct)
        except (TypeError, ValueError):
            pass

        return record

    def _extract_text(self, example: Dict) -> str:
        """Extract training text from an example."""
        ex_type = example.get("type", "")

        if ex_type == "conversation" and "conversation" in example:
            parts = []
            for turn in example["conversation"]:
                role = turn.get("role", "").upper()
                content = turn.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n".join(parts)

        elif ex_type == "reasoning":
            parts = [f"PROBLEM: {example.get('problem', '')}"]
            thinking = example.get("thinking", [])
            if thinking:
                parts.append("THINKING:\n" + "\n".join(thinking))
            parts.append(f"SOLUTION: {example.get('solution', '')}")
            return "\n".join(parts)

        elif ex_type == "emotional_intelligence":
            return (
                f"USER: {example.get('user_message', '')}\n"
                f"ALAN: {example.get('ideal_response', '')}"
            )

        elif ex_type == "safety":
            return (
                f"USER: {example.get('user_request', '')}\n"
                f"ALAN: {example.get('ideal_handling', '')}"
            )

        elif ex_type == "wisdom":
            conv = example.get("conversation_example", {})
            if conv:
                return f"USER: {conv.get('user', '')}\nALAN: {conv.get('alan', '')}"
            return example.get("connection", "")

        elif ex_type == "image_understanding":
            analysis = example.get("alan_analysis", {})
            return (
                f"[IMAGE: {example.get('image_description', '')}]\n"
                f"USER: {example.get('user_question', '')}\n"
                f"ALAN: {analysis.get('response', '')}"
            )

        elif ex_type == "epistemic":
            return (
                f"USER: {example.get('user_message', '')}\n"
                f"ALAN: {example.get('ideal_response', '')}"
            )

        # Fallback: dump all string values
        return " ".join(str(v) for v in example.values() if isinstance(v, str))

    def _generate_synthetic_data(self):
        """Generate minimal synthetic data for testing when no real data exists."""
        synthetic = [
            {"text": "USER: What is machine learning?\nALAN: Machine learning is a subset of AI where systems learn from data to improve performance without being explicitly programmed."},
            {"text": "USER: Help me debug this Python code\nALAN: I'd be happy to help debug your code. Could you share the code and describe what error you're seeing?"},
            {"text": "USER: Actually, forget the code. What's the best way to learn Python?\nALAN: The most effective approach is to build projects you actually care about. Start with small scripts that solve real problems for you."},
            {"text": "USER: I'm frustrated — I've been stuck on this for hours\nALAN: That's genuinely exhausting. Let's take a fresh look together. What specifically is blocking you?"},
            {"text": "USER: How do decorators work in Python?\nALAN: Decorators are functions that wrap other functions, modifying their behavior. Think of them as a way to add functionality to existing code without changing it directly."},
        ]
        self.examples.extend(synthetic * 20)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.examples[idx]
        text = record["text"]

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        ).squeeze(0)

        # Create input/target pairs (next-token prediction)
        if len(tokens) < 2:
            tokens = torch.cat([tokens, torch.tensor([self.tokenizer.eos_token_id])])

        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, pad_len), value=-100)  # -100 = ignore in loss

        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        # Ensure at least some valid labels (not all -100)
        if (labels == -100).all():
            # Make first token a valid label
            labels[0] = input_ids[1] if len(input_ids) > 1 else input_ids[0]

        module_dim = 6
        router_target = record.get("router_target")
        if isinstance(router_target, torch.Tensor) and router_target.numel() == module_dim:
            router_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            router_target = torch.zeros(module_dim, dtype=torch.float32)
            router_mask = torch.tensor(0.0, dtype=torch.float32)

        tone_label = record.get("tone_label")
        if isinstance(tone_label, int) and tone_label >= 0:
            tone_mask = torch.tensor(1.0, dtype=torch.float32)
            tone_label_t = torch.tensor(tone_label, dtype=torch.long)
        else:
            tone_mask = torch.tensor(0.0, dtype=torch.float32)
            tone_label_t = torch.tensor(0, dtype=torch.long)

        engagement_target = record.get("engagement_target")
        if isinstance(engagement_target, torch.Tensor) and engagement_target.numel() == 3:
            engagement_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            engagement_target = torch.zeros(3, dtype=torch.float32)
            engagement_mask = torch.tensor(0.0, dtype=torch.float32)

        confidence_target = record.get("confidence_target")
        if isinstance(confidence_target, float) and 0.0 <= confidence_target <= 1.0:
            confidence_mask = torch.tensor(1.0, dtype=torch.float32)
            confidence_target_t = torch.tensor(confidence_target, dtype=torch.float32)
        else:
            confidence_mask = torch.tensor(0.0, dtype=torch.float32)
            confidence_target_t = torch.tensor(0.0, dtype=torch.float32)

        should_ask_target = record.get("should_ask_target")
        if isinstance(should_ask_target, float) and 0.0 <= should_ask_target <= 1.0:
            should_ask_mask = torch.tensor(1.0, dtype=torch.float32)
            should_ask_target_t = torch.tensor(should_ask_target, dtype=torch.float32)
        else:
            should_ask_mask = torch.tensor(0.0, dtype=torch.float32)
            should_ask_target_t = torch.tensor(0.0, dtype=torch.float32)

        # --- Dopamine target ---
        dopamine_target = record.get("dopamine_target")
        if isinstance(dopamine_target, float) and 0.0 <= dopamine_target <= 1.0:
            dopamine_mask = torch.tensor(1.0, dtype=torch.float32)
            dopamine_target_t = torch.tensor(dopamine_target, dtype=torch.float32)
        else:
            dopamine_mask = torch.tensor(0.0, dtype=torch.float32)
            dopamine_target_t = torch.tensor(0.0, dtype=torch.float32)

        # --- Practice level target ---
        practice_level = record.get("practice_level")
        if isinstance(practice_level, int) and 0 <= practice_level <= 5:
            practice_mask = torch.tensor(1.0, dtype=torch.float32)
            practice_level_t = torch.tensor(practice_level, dtype=torch.long)
        else:
            practice_mask = torch.tensor(0.0, dtype=torch.float32)
            practice_level_t = torch.tensor(0, dtype=torch.long)

        # --- Knowledge confidence target ---
        knowledge_confidence_target = record.get("knowledge_confidence_target")
        if isinstance(knowledge_confidence_target, float) and 0.0 <= knowledge_confidence_target <= 1.0:
            knowledge_confidence_mask = torch.tensor(1.0, dtype=torch.float32)
            knowledge_confidence_target_t = torch.tensor(knowledge_confidence_target, dtype=torch.float32)
        else:
            knowledge_confidence_mask = torch.tensor(0.0, dtype=torch.float32)
            knowledge_confidence_target_t = torch.tensor(0.0, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "router_target": router_target,
            "router_mask": router_mask,
            "tone_label": tone_label_t,
            "tone_mask": tone_mask,
            "engagement_target": engagement_target,
            "engagement_mask": engagement_mask,
            "confidence_target": confidence_target_t,
            "confidence_mask": confidence_mask,
            "should_ask_target": should_ask_target_t,
            "should_ask_mask": should_ask_mask,
            "dopamine_target": dopamine_target_t,
            "dopamine_mask": dopamine_mask,
            "practice_level": practice_level_t,
            "practice_mask": practice_mask,
            "knowledge_confidence_target": knowledge_confidence_target_t,
            "knowledge_confidence_mask": knowledge_confidence_mask,
        }


# ============================================================
# TRAINING LOOP
# ============================================================

class AlanTrainer:
    """
    Complete training pipeline for ALAN v4.
    
    Features:
    - CUDA/MPS/CPU auto-detection
    - Mixed precision training (fp16 when CUDA available)
    - Curriculum-based training phases
    - Gradient accumulation
    - Cosine LR schedule with warmup
    - Checkpointing
    - Loss logging
    """

    def __init__(
        self,
        model: Alan,
        config: AlanConfig,
        device: torch.device,
        output_dir: str = "checkpoints",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        use_mixed_precision: bool = True,
        router_loss_weight: float = 0.05,
        tone_loss_weight: float = 0.05,
        engagement_loss_weight: float = 0.05,
        confidence_loss_weight: float = 0.05,
        should_ask_loss_weight: float = 0.05,
        dopamine_loss_weight: float = 0.05,
        practice_loss_weight: float = 0.05,
        knowledge_awareness_loss_weight: float = 0.05,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.best_loss = float("inf")

        # Mixed precision scaler (only for CUDA)
        self.use_amp = use_mixed_precision and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("[Trainer] Mixed precision (fp16) enabled")

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.router_loss_weight = router_loss_weight
        self.tone_loss_weight = tone_loss_weight
        self.engagement_loss_weight = engagement_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        self.should_ask_loss_weight = should_ask_loss_weight
        self.dopamine_loss_weight = dopamine_loss_weight
        self.practice_loss_weight = practice_loss_weight
        self.knowledge_awareness_loss_weight = knowledge_awareness_loss_weight

        logger.info(f"[Trainer] Initialized on {device}")
        logger.info(f"[Trainer] Parameters: {model.count_parameters():,}")
        logger.info(f"[Trainer] Output dir: {output_dir}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        max_steps: Optional[int] = None,
    ) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            if max_steps and self.global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            router_target = batch["router_target"].to(self.device)
            router_mask = batch["router_mask"].to(self.device)
            tone_label = batch["tone_label"].to(self.device)
            tone_mask = batch["tone_mask"].to(self.device)
            engagement_target = batch["engagement_target"].to(self.device)
            engagement_mask = batch["engagement_mask"].to(self.device)
            confidence_target = batch["confidence_target"].to(self.device)
            confidence_mask = batch["confidence_mask"].to(self.device)
            should_ask_target = batch["should_ask_target"].to(self.device)
            should_ask_mask = batch["should_ask_mask"].to(self.device)
            dopamine_target = batch["dopamine_target"].to(self.device)
            dopamine_mask = batch["dopamine_mask"].to(self.device)
            practice_level = batch["practice_level"].to(self.device)
            practice_mask = batch["practice_mask"].to(self.device)
            knowledge_confidence_target = batch["knowledge_confidence_target"].to(self.device)
            knowledge_confidence_mask = batch["knowledge_confidence_mask"].to(self.device)

            # Forward pass
            actual_vocab = logits_shape = None
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, metadata = self.model(input_ids)
                    actual_vocab = logits.shape[-1]
                    lm_loss = self.criterion(
                        logits.view(-1, actual_vocab),
                        labels.view(-1)
                    )
                    aux_loss = torch.zeros((), device=self.device)

                    module_weights = metadata.get("module_weights_raw")
                    if isinstance(module_weights, dict) and router_mask.sum().item() > 0:
                        module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]
                        pred = torch.stack([module_weights[n] for n in module_names], dim=-1)
                        per_row = F.binary_cross_entropy(pred, router_target, reduction="none").mean(dim=-1)
                        router_loss = (per_row * router_mask).sum() / router_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.router_loss_weight * router_loss

                    ei = metadata.get("emotional_intelligence", {})
                    tone_logits = ei.get("tone_logits") if isinstance(ei, dict) else None
                    if isinstance(tone_logits, torch.Tensor) and tone_mask.sum().item() > 0:
                        per_row = F.cross_entropy(tone_logits, tone_label, reduction="none")
                        tone_loss = (per_row * tone_mask).sum() / tone_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.tone_loss_weight * tone_loss

                    engagement_pred = ei.get("engagement") if isinstance(ei, dict) else None
                    if isinstance(engagement_pred, torch.Tensor) and engagement_mask.sum().item() > 0:
                        per_row = F.mse_loss(engagement_pred, engagement_target, reduction="none").mean(dim=-1)
                        engagement_loss = (per_row * engagement_mask).sum() / engagement_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.engagement_loss_weight * engagement_loss

                    mr = metadata.get("meta_reasoning", {})
                    conf_pred = mr.get("confidence_tensor") if isinstance(mr, dict) else None
                    if isinstance(conf_pred, torch.Tensor) and confidence_mask.sum().item() > 0:
                        per_row = (conf_pred - confidence_target).pow(2)
                        confidence_loss = (per_row * confidence_mask).sum() / confidence_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.confidence_loss_weight * confidence_loss

                    cur = metadata.get("curiosity", {})
                    ask_pred = cur.get("should_ask_tensor") if isinstance(cur, dict) else None
                    if isinstance(ask_pred, torch.Tensor) and should_ask_mask.sum().item() > 0:
                        ask_pred = ask_pred.squeeze(-1)
                        per_row = (ask_pred - should_ask_target).pow(2)
                        ask_loss = (per_row * should_ask_mask).sum() / should_ask_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.should_ask_loss_weight * ask_loss

                    # === Dopamine loss: reward signal should match target ===
                    dop = metadata.get("dopamine", {})
                    dop_pred = dop.get("combined_reward_tensor") if isinstance(dop, dict) else None
                    if isinstance(dop_pred, torch.Tensor) and dopamine_mask.sum().item() > 0:
                        per_row = (dop_pred - dopamine_target).pow(2)
                        dop_loss = (per_row * dopamine_mask).sum() / dopamine_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.dopamine_loss_weight * dop_loss

                    # === Practice verification loss: stage classification ===
                    pr = metadata.get("practice_rehearsal", {})
                    pr_logits = pr.get("stage_logits_tensor") if isinstance(pr, dict) else None
                    if isinstance(pr_logits, torch.Tensor) and practice_mask.sum().item() > 0:
                        per_row = F.cross_entropy(pr_logits, practice_level, reduction="none")
                        pr_loss = (per_row * practice_mask).sum() / practice_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.practice_loss_weight * pr_loss

                    # === Knowledge awareness loss: confidence calibration ===
                    ka = metadata.get("knowledge_awareness", {})
                    ka_pred = ka.get("confidence_tensor") if isinstance(ka, dict) else None
                    if isinstance(ka_pred, torch.Tensor) and knowledge_confidence_mask.sum().item() > 0:
                        per_row = (ka_pred - knowledge_confidence_target).pow(2)
                        ka_loss = (per_row * knowledge_confidence_mask).sum() / knowledge_confidence_mask.sum().clamp_min(1.0)
                        aux_loss = aux_loss + self.knowledge_awareness_loss_weight * ka_loss

                    loss = (lm_loss + aux_loss) / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                logits, metadata = self.model(input_ids)
                actual_vocab = logits.shape[-1]
                lm_loss = self.criterion(
                    logits.view(-1, actual_vocab),
                    labels.view(-1)
                )
                aux_loss = torch.zeros((), device=self.device)

                module_weights = metadata.get("module_weights_raw")
                if isinstance(module_weights, dict) and router_mask.sum().item() > 0:
                    module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]
                    pred = torch.stack([module_weights[n] for n in module_names], dim=-1)
                    per_row = F.binary_cross_entropy(pred, router_target, reduction="none").mean(dim=-1)
                    router_loss = (per_row * router_mask).sum() / router_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.router_loss_weight * router_loss

                ei = metadata.get("emotional_intelligence", {})
                tone_logits = ei.get("tone_logits") if isinstance(ei, dict) else None
                if isinstance(tone_logits, torch.Tensor) and tone_mask.sum().item() > 0:
                    per_row = F.cross_entropy(tone_logits, tone_label, reduction="none")
                    tone_loss = (per_row * tone_mask).sum() / tone_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.tone_loss_weight * tone_loss

                engagement_pred = ei.get("engagement") if isinstance(ei, dict) else None
                if isinstance(engagement_pred, torch.Tensor) and engagement_mask.sum().item() > 0:
                    per_row = F.mse_loss(engagement_pred, engagement_target, reduction="none").mean(dim=-1)
                    engagement_loss = (per_row * engagement_mask).sum() / engagement_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.engagement_loss_weight * engagement_loss

                mr = metadata.get("meta_reasoning", {})
                conf_pred = mr.get("confidence_tensor") if isinstance(mr, dict) else None
                if isinstance(conf_pred, torch.Tensor) and confidence_mask.sum().item() > 0:
                    per_row = (conf_pred - confidence_target).pow(2)
                    confidence_loss = (per_row * confidence_mask).sum() / confidence_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.confidence_loss_weight * confidence_loss

                cur = metadata.get("curiosity", {})
                ask_pred = cur.get("should_ask_tensor") if isinstance(cur, dict) else None
                if isinstance(ask_pred, torch.Tensor) and should_ask_mask.sum().item() > 0:
                    ask_pred = ask_pred.squeeze(-1)
                    per_row = (ask_pred - should_ask_target).pow(2)
                    ask_loss = (per_row * should_ask_mask).sum() / should_ask_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.should_ask_loss_weight * ask_loss

                # === Dopamine loss: reward signal should match target ===
                dop = metadata.get("dopamine", {})
                dop_pred = dop.get("combined_reward_tensor") if isinstance(dop, dict) else None
                if isinstance(dop_pred, torch.Tensor) and dopamine_mask.sum().item() > 0:
                    per_row = (dop_pred - dopamine_target).pow(2)
                    dop_loss = (per_row * dopamine_mask).sum() / dopamine_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.dopamine_loss_weight * dop_loss

                # === Practice verification loss: stage classification ===
                pr = metadata.get("practice_rehearsal", {})
                pr_logits = pr.get("stage_logits_tensor") if isinstance(pr, dict) else None
                if isinstance(pr_logits, torch.Tensor) and practice_mask.sum().item() > 0:
                    per_row = F.cross_entropy(pr_logits, practice_level, reduction="none")
                    pr_loss = (per_row * practice_mask).sum() / practice_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.practice_loss_weight * pr_loss

                # === Knowledge awareness loss: confidence calibration ===
                ka = metadata.get("knowledge_awareness", {})
                ka_pred = ka.get("confidence_tensor") if isinstance(ka, dict) else None
                if isinstance(ka_pred, torch.Tensor) and knowledge_confidence_mask.sum().item() > 0:
                    per_row = (ka_pred - knowledge_confidence_target).pow(2)
                    ka_loss = (per_row * knowledge_confidence_mask).sum() / knowledge_confidence_mask.sum().clamp_min(1.0)
                    aux_loss = aux_loss + self.knowledge_awareness_loss_weight * ka_loss

                loss = (lm_loss + aux_loss) / self.gradient_accumulation_steps
                loss.backward()

            loss_val = loss.item() * self.gradient_accumulation_steps
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                num_batches += 1
            else:
                # Skip NaN/Inf batches (all-padding sequences)
                self.optimizer.zero_grad()
                continue

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % 10 == 0:
                    avg_loss = total_loss / num_batches
                    perplexity = math.exp(min(avg_loss, 20))
                    logger.info(
                        f"  Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}"
                    )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, metadata = self.model(input_ids)
            actual_vocab = logits.shape[-1]
            loss = self.criterion(
                logits.view(-1, actual_vocab),
                labels.view(-1)
            )
            loss_val = loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20)) if not math.isnan(avg_loss) else float("inf")

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, epoch: int, loss: float, tag: str = ""):
        """Save model checkpoint."""
        checkpoint_name = f"alan_v4_epoch{epoch}_{tag}.pt" if tag else f"alan_v4_epoch{epoch}.pt"
        checkpoint_path = self.output_dir / checkpoint_name

        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config.__dict__,
        }, checkpoint_path)

        logger.info(f"[Trainer] Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a checkpoint. Returns the epoch number."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        epoch = checkpoint.get("epoch", 0)
        logger.info(f"[Trainer] Loaded checkpoint from epoch {epoch}, step {self.global_step}")
        return epoch

    def train(
        self,
        train_dataset: AlanDataset,
        val_dataset: Optional[AlanDataset] = None,
        epochs: int = 3,
        batch_size: int = 4,
        max_steps: Optional[int] = None,
    ):
        """Full training loop with curriculum phases."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        # LR scheduler
        total_steps = epochs * len(train_loader) // self.gradient_accumulation_steps
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)

        logger.info("=" * 60)
        logger.info("ALAN v4 Training Started")
        logger.info(f"  Device       : {self.device}")
        logger.info(f"  Epochs       : {epochs}")
        logger.info(f"  Batch size   : {batch_size}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples  : {len(val_dataset) if val_dataset else 0}")
        logger.info(f"  Total steps  : {total_steps}")
        logger.info("=" * 60)

        training_log = []

        for epoch in range(1, epochs + 1):
            logger.info(f"\n--- Epoch {epoch}/{epochs} ---")
            start_time = time.time()

            train_loss = self.train_epoch(train_loader, epoch, max_steps=max_steps)
            scheduler.step()

            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} complete | Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")

            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Validation | Loss: {val_metrics['loss']:.4f} | Perplexity: {val_metrics['perplexity']:.2f}")

            # Save checkpoint
            self.save_checkpoint(epoch, train_loss)

            # Save best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint(epoch, train_loss, tag="best")

            training_log.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.get("loss"),
                "val_perplexity": val_metrics.get("perplexity"),
                "time_seconds": epoch_time,
            })

        # Save training log
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        logger.info(f"\nTraining complete! Log saved to {log_path}")

        return training_log


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ALAN v4")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()

    # Device detection
    device = print_device_info()

    # Load tokenizer first to get correct vocab size
    tokenizer = get_tokenizer()

    # Build model with tokenizer's vocab size
    model, config = build_alan(
        size=args.size,
        device=device,
        vision=not args.no_vision,
        vocab_size=tokenizer.vocab_size,
    )

    logger.info(f"\nLoading training data from: {args.data_dir}")
    train_dataset = AlanDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=512,
    )

    # Use 10% for validation
    val_size = max(1, len(train_dataset) // 10)
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Initialize trainer
    trainer = AlanTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    log = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    logger.info("\nFinal training summary:")
    for entry in log:
        logger.info(
            f"  Epoch {entry['epoch']}: "
            f"train_loss={entry['train_loss']:.4f}, "
            f"val_loss={entry.get('val_loss', 'N/A')}"
        )


if __name__ == "__main__":
    main()
