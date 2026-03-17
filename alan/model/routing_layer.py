"""
ALAN v4 — Routing Layer
ECONX GROUP (PTY) LTD

Task classification and module routing.
Re-exports TaskRouter from core_transformer
for use as a standalone module per the project structure spec.
"""

from .core_transformer import TaskRouter

__all__ = ["TaskRouter"]
