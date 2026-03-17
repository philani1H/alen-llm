"""
ALAN v4 — Modular Attention Module
ECONX GROUP (PTY) LTD

Specialized attention heads per cognitive module.
Re-exports ModularMultiHeadAttention from core_transformer
for use as a standalone module per the project structure spec.
"""

from .core_transformer import ModularMultiHeadAttention, RotaryPositionalEncoding

__all__ = ["ModularMultiHeadAttention", "RotaryPositionalEncoding"]
