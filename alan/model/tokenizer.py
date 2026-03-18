"""
ALAN v4 — Custom BPE Tokenizer
ECONX GROUP (PTY) LTD

Custom Byte-Pair Encoding tokenizer built from scratch for ALAN.
No dependency on OpenAI, GPT-2, or any third-party tokenizer.

This is ALAN's own tokenizer — part of building a fully custom
generative AI model from the ground up.

Features:
- Byte-level BPE (handles any Unicode text)
- Special tokens for scratchpad, system, and role markers
- Trainable on custom corpora
- Compatible with ALAN's extended vocabulary (vocab_size + 100 special tokens)
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, OrderedDict

logger = logging.getLogger(__name__)


# ============================================================
# SPECIAL TOKENS
# ============================================================

SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|eos|>": 1,
    "<|bos|>": 2,
    "<|unk|>": 3,
    "<|sep|>": 4,
    "<|think|>": 5,       # Scratchpad start
    "<|/think|>": 6,      # Scratchpad end
    "<|system|>": 7,      # System message marker
    "<|user|>": 8,        # User turn marker
    "<|alan|>": 9,        # ALAN turn marker
    "<|image|>": 10,      # Image token marker
    "<|newline|>": 11,    # Explicit newline
}

NUM_SPECIAL_TOKENS = 100  # Reserve 100 IDs for special tokens


# ============================================================
# BYTE-LEVEL BPE TOKENIZER
# ============================================================

class AlanTokenizer:
    """
    ALAN's custom BPE tokenizer.

    Built from scratch — no GPT-2, no OpenAI, no HuggingFace dependency.

    The tokenizer works at the byte level, meaning it can handle any
    Unicode text. BPE merges are learned from training data to create
    an efficient subword vocabulary.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens or SPECIAL_TOKENS

        # Base vocabulary: 256 byte values + special tokens
        self.byte_vocab = {bytes([i]): i + NUM_SPECIAL_TOKENS for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_vocab.items()}

        # BPE merges (learned from training data or loaded from file)
        self.merges: List[Tuple[bytes, bytes]] = []
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {}

        # Full vocabulary: special tokens + byte vocab + BPE merges
        self.vocab: Dict[bytes, int] = {}
        self.decoder: Dict[int, bytes] = {}

        # Build initial vocabulary
        self._build_vocab()

        # Properties for compatibility
        self.pad_token_id = self.special_tokens["<|pad|>"]
        self.eos_token_id = self.special_tokens["<|eos|>"]
        self.bos_token_id = self.special_tokens["<|bos|>"]
        self.unk_token_id = self.special_tokens["<|unk|>"]
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.vocab_size = len(self.vocab) + len(self.special_tokens)

    def _build_vocab(self):
        """Build the full vocabulary from special tokens + bytes + merges."""
        self.vocab = {}
        self.decoder = {}

        # Special tokens get IDs 0 to NUM_SPECIAL_TOKENS-1
        for token, idx in self.special_tokens.items():
            token_bytes = token.encode("utf-8")
            self.vocab[token_bytes] = idx
            self.decoder[idx] = token_bytes

        # Byte-level tokens get IDs starting at NUM_SPECIAL_TOKENS
        for byte_val, idx in self.byte_vocab.items():
            self.vocab[byte_val] = idx
            self.decoder[idx] = byte_val

        # BPE merge tokens
        next_id = NUM_SPECIAL_TOKENS + 256
        for pair in self.merges:
            merged = pair[0] + pair[1]
            if merged not in self.vocab:
                self.vocab[merged] = next_id
                self.decoder[next_id] = merged
                next_id += 1

        self.vocab_size = next_id

    def _get_pairs(self, tokens: List[bytes]) -> Counter:
        """Get frequency of adjacent token pairs."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens: List[bytes], pair: Tuple[bytes, bytes]) -> List[bytes]:
        """Merge all occurrences of a pair in the token list."""
        merged = pair[0] + pair[1]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, texts: List[str], num_merges: Optional[int] = None):
        """
        Train BPE merges from a list of texts.

        Args:
            texts: training texts
            num_merges: number of merge operations (default: vocab_size - 256 - special)
        """
        if num_merges is None:
            num_merges = self.target_vocab_size - 256 - NUM_SPECIAL_TOKENS

        logger.info(f"[Tokenizer] Training BPE with {num_merges} merges on {len(texts)} texts")

        # Convert all texts to byte sequences
        all_tokens = []
        for text in texts:
            byte_seq = [bytes([b]) for b in text.encode("utf-8")]
            all_tokens.append(byte_seq)

        self.merges = []
        self.merge_ranks = {}

        for merge_idx in range(num_merges):
            # Count all pairs across all sequences
            pair_counts = Counter()
            for tokens in all_tokens:
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] += 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # Record the merge
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = merge_idx

            # Apply merge to all sequences
            all_tokens = [self._merge_pair(seq, best_pair) for seq in all_tokens]

            if (merge_idx + 1) % 1000 == 0:
                logger.info(f"  Merge {merge_idx + 1}/{num_merges}")

        # Rebuild vocabulary with new merges
        self._build_vocab()
        logger.info(f"[Tokenizer] Training complete. Vocab size: {self.vocab_size}")

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
    ):
        """
        Encode text to token IDs.

        Args:
            text: input text string
            max_length: maximum sequence length
            truncation: whether to truncate to max_length
            return_tensors: if "pt", return PyTorch tensor
            add_special_tokens: if True, add BOS/EOS tokens

        Returns:
            List of token IDs, or tensor if return_tensors="pt"
        """
        # Handle special token markers in text
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        # Check for special tokens in text
        parts = self._split_special_tokens(text)

        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.special_tokens[part])
            else:
                # BPE encode the text part
                ids = self._bpe_encode(part)
                token_ids.extend(ids)

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        # Truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Return format
        if return_tensors == "pt":
            import torch
            return torch.tensor([token_ids], dtype=torch.long)

        return token_ids

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: list or tensor of token IDs
            skip_special_tokens: if True, omit special tokens from output

        Returns:
            Decoded text string
        """
        # Handle tensor input
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        byte_parts = []
        for tid in token_ids:
            if tid in self.decoder:
                token_bytes = self.decoder[tid]
                # Check if it's a special token
                try:
                    token_str = token_bytes.decode("utf-8")
                    if skip_special_tokens and token_str in self.special_tokens:
                        continue
                except UnicodeDecodeError:
                    pass
                byte_parts.append(token_bytes)
            # Skip unknown IDs silently

        full_bytes = b"".join(byte_parts)
        try:
            return full_bytes.decode("utf-8", errors="replace")
        except Exception:
            return full_bytes.decode("latin-1", errors="replace")

    def _bpe_encode(self, text: str) -> List[int]:
        """Apply BPE encoding to a text string."""
        if not text:
            return []

        # Start with byte-level tokens
        tokens = [bytes([b]) for b in text.encode("utf-8")]

        # Apply merges in order
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
            if len(tokens) <= 1:
                break

        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Fallback to individual bytes
                for b in token:
                    byte_key = bytes([b])
                    ids.append(self.byte_vocab.get(byte_key, self.unk_token_id))

        return ids

    def _split_special_tokens(self, text: str) -> List[str]:
        """Split text around special token markers."""
        # Build regex pattern for special tokens
        special_pattern = "|".join(
            re.escape(token) for token in sorted(self.special_tokens.keys(), key=len, reverse=True)
        )
        if not special_pattern:
            return [text] if text else []

        parts = re.split(f"({special_pattern})", text)
        return [p for p in parts if p]

    def save(self, path: str):
        """Save tokenizer to a directory."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save merges
        merges_data = []
        for pair in self.merges:
            merges_data.append({
                "a": list(pair[0]),
                "b": list(pair[1]),
            })

        with open(save_dir / "merges.json", "w") as f:
            json.dump(merges_data, f)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
        }
        with open(save_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"[Tokenizer] Saved to {save_dir}")

    @classmethod
    def load(cls, path: str) -> "AlanTokenizer":
        """Load tokenizer from a directory."""
        load_dir = Path(path)

        # Load config
        with open(load_dir / "tokenizer_config.json", "r") as f:
            config = json.load(f)

        tokenizer = cls(
            vocab_size=config["target_vocab_size"],
            special_tokens=config.get("special_tokens", SPECIAL_TOKENS),
        )

        # Load merges
        merges_path = load_dir / "merges.json"
        if merges_path.exists():
            with open(merges_path, "r") as f:
                merges_data = json.load(f)

            tokenizer.merges = [
                (bytes(m["a"]), bytes(m["b"]))
                for m in merges_data
            ]
            tokenizer.merge_ranks = {
                pair: i for i, pair in enumerate(tokenizer.merges)
            }
            tokenizer._build_vocab()

        logger.info(f"[Tokenizer] Loaded from {load_dir} (vocab_size={tokenizer.vocab_size})")
        return tokenizer

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (
            f"AlanTokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, "
            f"special_tokens={len(self.special_tokens)})"
        )


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_tokenizer(
    tokenizer_path: Optional[str] = None,
    vocab_size: int = 50257,
    train_texts: Optional[List[str]] = None,
) -> AlanTokenizer:
    """
    Get ALAN's tokenizer. Loads from disk if available, otherwise creates new.

    Args:
        tokenizer_path: path to saved tokenizer directory
        vocab_size: target vocabulary size
        train_texts: if provided, train on these texts

    Returns:
        AlanTokenizer instance
    """
    # Try loading saved tokenizer
    if tokenizer_path:
        path = Path(tokenizer_path)
        if (path / "tokenizer_config.json").exists():
            try:
                return AlanTokenizer.load(tokenizer_path)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")

    # Create new tokenizer
    tokenizer = AlanTokenizer(vocab_size=vocab_size)

    # Train if texts provided
    if train_texts:
        # Limit merges for reasonable training time
        num_merges = min(vocab_size - 256 - NUM_SPECIAL_TOKENS, 5000)
        tokenizer.train(train_texts, num_merges=num_merges)

    logger.info(f"[Tokenizer] Created new AlanTokenizer: {tokenizer}")
    return tokenizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the tokenizer
    print("=== ALAN Custom BPE Tokenizer Test ===\n")

    tokenizer = AlanTokenizer(vocab_size=50257)

    # Test basic encoding/decoding
    test_texts = [
        "Hello, how are you?",
        "ALAN is a custom generative AI built from scratch.",
        "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "What is 7 * 8? The answer is 56.",
        "USER: Help me with Python\nALAN: I'd be happy to help!",
    ]

    print("Testing encode/decode (byte-level, no BPE merges):\n")
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        match = decoded == text
        print(f"  Input:   {text[:60]}...")
        print(f"  Tokens:  {len(ids)} tokens")
        print(f"  Match:   {'OK' if match else 'MISMATCH'}")
        print()

    # Test with PyTorch tensors
    print("Testing tensor output:")
    tensor_ids = tokenizer.encode("Hello world", return_tensors="pt")
    print(f"  Tensor shape: {tensor_ids.shape}")
    print(f"  Tensor dtype: {tensor_ids.dtype}")
    print()

    # Train on test data
    print("Testing BPE training:")
    tokenizer.train(test_texts * 10, num_merges=50)
    print(f"  Vocab size after training: {tokenizer.vocab_size}")

    for text in test_texts[:2]:
        ids_after = tokenizer.encode(text)
        decoded = tokenizer.decode(ids_after)
        print(f"  '{text[:40]}...' → {len(ids_after)} tokens (was byte-level)")
        assert decoded == text, f"Decode mismatch: {decoded}"

    print("\nAll tokenizer tests passed!")
