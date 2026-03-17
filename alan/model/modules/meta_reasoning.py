"""
ALAN v4 — Meta-Reasoning / Self-Discovery Module
ECONX GROUP (PTY) LTD

The 'inner critic' — validates reasoning, detects contradictions,
and scores confidence. Runs AFTER initial reasoning but BEFORE
output generation.

Re-exports MetaReasoning from emotional_intelligence
for use as a standalone module per the project structure spec.
"""

from .emotional_intelligence import MetaReasoning

__all__ = ["MetaReasoning"]
