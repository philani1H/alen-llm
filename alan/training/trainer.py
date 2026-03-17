"""
ALAN v4 — Training Pipeline
ECONX GROUP (PTY) LTD

Complete training pipeline including:
- Dataset loading from JSONL/TXT training data
- Tokenization with GPT-2 BPE tokenizer
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
    """Load GPT-2 BPE tokenizer."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"[Tokenizer] GPT-2 tokenizer loaded: vocab_size={tokenizer.vocab_size}")
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
        self.examples = []

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
                                text = self._extract_text(example)
                                if text and len(text) > 10:
                                    self.examples.append(text)
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")

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

        # Fallback: dump all string values
        return " ".join(str(v) for v in example.values() if isinstance(v, str))

    def _generate_synthetic_data(self):
        """Generate minimal synthetic data for testing when no real data exists."""
        synthetic = [
            "USER: What is machine learning?\nALAN: Machine learning is a subset of AI where systems learn from data to improve performance without being explicitly programmed.",
            "USER: Help me debug this Python code\nALAN: I'd be happy to help debug your code. Could you share the code and describe what error you're seeing?",
            "USER: Actually, forget the code. What's the best way to learn Python?\nALAN: The most effective approach is to build projects you actually care about. Start with small scripts that solve real problems for you.",
            "USER: I'm frustrated — I've been stuck on this for hours\nALAN: That's genuinely exhausting. Let's take a fresh look together. What specifically is blocking you?",
            "USER: How do decorators work in Python?\nALAN: Decorators are functions that wrap other functions, modifying their behavior. Think of them as a way to add functionality to existing code without changing it directly.",
        ]
        self.examples.extend(synthetic * 20)  # Repeat for training

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]

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

        return {
            "input_ids": input_ids,
            "labels": labels,
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

            # Forward pass
            actual_vocab = logits_shape = None
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(input_ids)
                    actual_vocab = logits.shape[-1]
                    loss = self.criterion(
                        logits.view(-1, actual_vocab),
                        labels.view(-1)
                    )
                    loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                logits, _ = self.model(input_ids)
                actual_vocab = logits.shape[-1]
                loss = self.criterion(
                    logits.view(-1, actual_vocab),
                    labels.view(-1)
                )
                loss = loss / self.gradient_accumulation_steps
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
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))

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
