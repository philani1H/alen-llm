"""
ALAN v4 — Topic Tracking Evaluation
ECONX GROUP (PTY) LTD

Evaluates ALAN's Attention-to-Context system:
- Topic extraction accuracy
- Topic shift detection accuracy
- Explicit reference detection
- Recency decay correctness
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.memory.context_tracker import AttentionToContext

logger = logging.getLogger(__name__)


class TopicTrackingEval:
    """
    Evaluates the Attention-to-Context topic tracking system.

    Test scenarios:
    1. Simple topic continuity
    2. Explicit topic shifts ("actually...", "by the way...")
    3. Explicit back-references ("going back to...")
    4. Multi-topic conversations
    5. Recency decay behavior
    """

    def __init__(self):
        self.atc = AttentionToContext()

    def evaluate_topic_shifts(self) -> Dict:
        """Test topic shift detection accuracy."""
        self.atc.reset()

        test_cases = [
            ("Help me with Python decorators", False),
            ("How do I use @property?", False),
            ("Actually, I need help with my resume", True),
            ("What skills should I highlight?", False),
            ("By the way, what's the weather like?", True),
            ("Is it going to rain tomorrow?", False),
        ]

        correct = 0
        total = len(test_cases)

        for message, expected_shift in test_cases:
            ctx = self.atc.process_user_message(message)
            detected_shift = ctx["topic_shift"]
            if detected_shift == expected_shift:
                correct += 1

        return {
            "total_tests": total,
            "correct": correct,
            "accuracy": correct / total,
        }

    def evaluate_back_references(self) -> Dict:
        """Test explicit back-reference detection."""
        self.atc.reset()

        # Build up conversation history
        self.atc.process_user_message("Help me with Python decorators")
        self.atc.process_user_message("How do I use @property?")
        self.atc.process_user_message("Actually, I need help with my resume")
        self.atc.process_user_message("What skills should I highlight?")

        # Test back-references
        test_refs = [
            ("Going back to decorators — can they take arguments?", True),
            ("Tell me more about resume formatting", False),
            ("As I mentioned earlier about Python", True),
            ("What's the best IDE?", False),
        ]

        correct = 0
        for message, expected_ref in test_refs:
            ctx = self.atc.process_user_message(message)
            if ctx["explicit_reference"] == expected_ref:
                correct += 1

        return {
            "total_tests": len(test_refs),
            "correct": correct,
            "accuracy": correct / len(test_refs),
        }

    def evaluate_recency_decay(self) -> Dict:
        """Test that topic recency scores decay over time."""
        self.atc.reset()

        # Add a topic
        self.atc.process_user_message("Help me with Python decorators")
        initial_topics = self.atc.get_active_topics()

        # Add several more messages on different topics
        self.atc.process_user_message("What about my resume?")
        self.atc.process_user_message("Tell me about machine learning")
        self.atc.process_user_message("How does React work?")
        self.atc.process_user_message("What's the weather?")

        after_topics = self.atc.get_active_topics()

        # The first topic should have decayed
        first_topic_keys = list(initial_topics.keys())
        first_topic_decayed = True
        if first_topic_keys:
            for key in first_topic_keys:
                if key in after_topics:
                    if after_topics[key] >= initial_topics[key]:
                        first_topic_decayed = False

        return {
            "initial_active_topics": len(initial_topics),
            "after_active_topics": len(after_topics),
            "first_topic_decayed": first_topic_decayed,
            "decay_working": first_topic_decayed,
        }

    def run_all(self) -> Dict:
        """Run all topic tracking evaluations."""
        return {
            "topic_shifts": self.evaluate_topic_shifts(),
            "back_references": self.evaluate_back_references(),
            "recency_decay": self.evaluate_recency_decay(),
        }
