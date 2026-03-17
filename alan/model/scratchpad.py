"""
ALAN v4 — Internal Scratchpad (Thinking Token Mechanism)
ECONX GROUP (PTY) LTD

Internal chain-of-thought scratchpad that allows ALAN to
'think before answering' by generating internal reasoning tokens
that are not shown to the user.

Re-exports ScratchpadMechanism from reasoning_engine
for use as a standalone module per the project structure spec.
"""

from .modules.reasoning_engine import ScratchpadMechanism

__all__ = ["ScratchpadMechanism"]
