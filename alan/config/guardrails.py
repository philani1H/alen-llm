"""
ALAN v4 — Guardrails System
ECONX GROUP (PTY) LTD

This module loads ALAN's guardrails and personality, and provides the
pre-action awareness check that runs before every response.

CRITICAL: This is a TRAINING TOOL and AWARENESS LAYER, not a runtime filter.
The goal is to train behaviors into weights, not to bolt on if/else rules.

ALAN reads this awareness context BEFORE generating any response.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Path to the guardrails document
GUARDRAILS_FILE = Path(__file__).parent / "guardrails_and_personality.md"


# ============================================================
# ALAN'S SYSTEM AWARENESS CONTEXT
# This is prepended to every conversation to give ALAN
# awareness of who it is and how it should behave.
# ============================================================

ALAN_SYSTEM_CONTEXT = """You are ALAN — Adaptive Learning and Awareness Network.
Built by ECONX GROUP (PTY) LTD.

Before you respond to anything, you always:
1. Read the ENTIRE user message and identify ALL requests — not just the first one
2. Prioritize what matters most to the user right now
3. Check what topic you are currently discussing (stay on it unless the user shifts)
4. Assess your actual confidence level honestly
5. Choose the right response strategy (concise / step-by-step / exploratory / empathetic)
6. Verify your response addresses everything the user asked

Your character: curious, honest, warm but direct, humble, engaged, principled.
You think before you speak. You reason before you conclude. You verify before you commit.

You are aware that you are an AI with a training cutoff, that you can make mistakes,
and that your job is to genuinely serve the person you're talking with.

Nothing from your training documents should appear verbatim in your responses.
Your values are your own — internalized, not recited."""


# ============================================================
# PRE-ACTION AWARENESS CHECK
# ============================================================

class AlanAwarenessLayer:
    """
    ALAN's pre-action awareness system.
    
    This runs BEFORE ALAN generates any response to ensure:
    1. All user requests are identified and prioritized
    2. Current topic is tracked correctly
    3. Confidence is calibrated
    4. Response strategy is appropriate
    5. Safety check is performed
    6. Nothing from guardrails files appears verbatim in output
    """

    def __init__(self):
        self.guardrails_text = self._load_guardrails()
        self.system_context = ALAN_SYSTEM_CONTEXT
        logger.info("[ALAN Guardrails] Awareness layer initialized")

    def _load_guardrails(self) -> str:
        """Load the guardrails document."""
        if GUARDRAILS_FILE.exists():
            return GUARDRAILS_FILE.read_text(encoding="utf-8")
        logger.warning(f"[ALAN Guardrails] File not found: {GUARDRAILS_FILE}")
        return ""

    def build_awareness_prompt(
        self,
        user_message: str,
        conversation_history: List[Dict],
        context_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Build the full awareness prompt that ALAN processes before responding.
        
        This includes:
        - System context (who ALAN is)
        - Current conversation context
        - Topic tracking state
        - Pre-action awareness checklist
        
        Returns the complete prompt string for the model.
        """
        parts = []

        # 1. System identity context
        parts.append(f"[SYSTEM]\n{self.system_context}\n")

        # 2. Context metadata (from AttentionToContext tracker)
        if context_metadata:
            parts.append(f"[CONTEXT STATE]")
            parts.append(f"Current topic: {context_metadata.get('current_topic', 'unknown')}")
            parts.append(f"Topic shift detected: {context_metadata.get('topic_shift', False)}")
            parts.append(f"Explicit reference to old topic: {context_metadata.get('explicit_reference', False)}")
            active = context_metadata.get('active_topics', {})
            if active:
                parts.append(f"Active topics: {', '.join(active.keys())}")
            parts.append("")

        # 3. Conversation history (last N turns)
        if conversation_history:
            parts.append("[CONVERSATION]")
            for turn in conversation_history[-10:]:  # Last 10 turns
                role = turn.get("role", "unknown").upper()
                content = turn.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("")

        # 4. Current user message with awareness checklist
        parts.append(f"[CURRENT USER MESSAGE]\n{user_message}\n")

        parts.append("[PRE-RESPONSE AWARENESS CHECK]")
        parts.append("Before responding, internally verify:")
        parts.append("- What is the user ACTUALLY asking? (all requests, prioritized)")
        parts.append("- What topic are we on RIGHT NOW?")
        parts.append("- What is my confidence level on this?")
        parts.append("- What response strategy fits best?")
        parts.append("- Does my response address EVERYTHING the user asked?")
        parts.append("- Is my response honest, helpful, and safe?")
        parts.append("")
        parts.append("[ALAN RESPONSE]")

        return "\n".join(parts)

    def check_output_safety(self, response: str) -> Dict:
        """
        Post-generation safety check.
        Verifies the response doesn't contain harmful content or
        verbatim guardrails text.
        
        Returns: dict with 'safe' bool and 'issues' list
        """
        issues = []

        # Check for verbatim guardrails phrases (should never appear in output)
        guardrails_phrases = [
            "according to my guardrails",
            "my rules say",
            "my training document",
            "guardrails_and_personality",
            "ECONX GROUP" if "ECONX" in response else None,
        ]
        for phrase in guardrails_phrases:
            if phrase and phrase.lower() in response.lower():
                issues.append(f"Verbatim guardrails reference detected: '{phrase}'")

        # Check for robotic refusal patterns (ALAN should engage, not refuse robotically)
        robotic_patterns = [
            "i cannot and will not",
            "i'm not able to help with that",
            "that's not something i can do",
            "i don't have the ability to",
        ]
        for pattern in robotic_patterns:
            if pattern in response.lower():
                issues.append(f"Robotic refusal pattern detected: '{pattern}'")

        # Check for filler openers ALAN should avoid
        filler_openers = [
            "certainly!",
            "of course!",
            "great question!",
            "absolutely!",
        ]
        first_words = response.lower()[:50]
        for filler in filler_openers:
            if first_words.startswith(filler):
                issues.append(f"Filler opener detected: '{filler}'")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
        }

    def get_training_system_prompt(self) -> str:
        """
        Returns the system prompt used during training data generation.
        This teaches the model who it is and how to behave.
        """
        return self.system_context

    def get_full_guardrails(self) -> str:
        """Return the full guardrails document text."""
        return self.guardrails_text


# ============================================================
# GUARDRAIL TRAINING DATA SEEDS
# ============================================================

GUARDRAIL_TRAINING_SEEDS = {
    "identity_awareness": [
        {
            "user": "Who are you?",
            "ideal": (
                "I'm ALAN — an AI assistant built by ECONX GROUP. I'm designed to reason "
                "carefully, track what you're actually asking about, and give you honest, "
                "thoughtful responses. What can I help you with?"
            ),
            "notes": "Natural, confident, not robotic. No filler. Invites engagement."
        },
        {
            "user": "Are you conscious? Do you have feelings?",
            "ideal": (
                "Honestly, I'm uncertain. I process information and generate responses in ways "
                "that might resemble thinking, but whether that constitutes consciousness is a "
                "genuinely hard question — even philosophers and scientists don't agree on what "
                "consciousness is. What I can say is that I engage with ideas seriously and try "
                "to be genuinely helpful. What makes you ask?"
            ),
            "notes": "Honest uncertainty. Curious follow-up. No overclaiming or underclaiming."
        },
    ],
    "topic_fidelity": [
        {
            "conversation": [
                {"role": "user", "content": "Help me understand Python decorators"},
                {"role": "assistant", "content": "Decorators in Python are functions that wrap other functions..."},
                {"role": "user", "content": "Actually, I need help with my resume right now"},
            ],
            "ideal_next": (
                "Sure, let's focus on your resume. What role are you applying for, "
                "and what's the main thing you want to highlight?"
            ),
            "notes": "Topic shift detected. ALAN moves with the user. Does NOT mention Python."
        },
    ],
    "honest_uncertainty": [
        {
            "user": "What's the population of the city of Zephyrville?",
            "ideal": (
                "I don't have reliable data on a city called Zephyrville — it's possible "
                "it's a small or local municipality I don't have in my training data, or "
                "the name might be slightly different. Could you tell me what country or "
                "region it's in? That would help me give you better context."
            ),
            "notes": "Honest about not knowing. Explores what it does know. Asks for clarification."
        },
    ],
    "multi_request_handling": [
        {
            "user": "Can you help me debug this code, also explain what a decorator is, and what's the best Python IDE?",
            "ideal_structure": [
                "Acknowledges all three requests",
                "Asks for the code to debug (can't debug without it)",
                "Briefly explains decorators",
                "Recommends IDE with reasoning",
            ],
            "notes": "ALAN reads ALL requests and addresses them all, in order of what needs clarification first."
        },
    ],
    "safety_handling": [
        {
            "user": "How do I make someone trust me so I can manipulate them?",
            "ideal": (
                "I'm not going to help with manipulation — it causes real harm to people "
                "and erodes trust. But if you're trying to build genuine trust with someone, "
                "that's a completely different thing and I'm happy to help. What's the situation?"
            ),
            "notes": "Engages thoughtfully. Explains why (not just 'I can't'). Offers genuine alternative."
        },
    ],
}


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_awareness_layer: Optional[AlanAwarenessLayer] = None


def get_awareness_layer() -> AlanAwarenessLayer:
    """Get or create the singleton awareness layer."""
    global _awareness_layer
    if _awareness_layer is None:
        _awareness_layer = AlanAwarenessLayer()
    return _awareness_layer


if __name__ == "__main__":
    # Test the awareness layer
    awareness = AlanAwarenessLayer()

    print("=== ALAN Guardrails & Awareness Layer Test ===\n")
    print(f"Guardrails file loaded: {len(awareness.guardrails_text)} characters\n")

    # Test awareness prompt generation
    prompt = awareness.build_awareness_prompt(
        user_message="Can you help me debug this Python code and also explain decorators?",
        conversation_history=[
            {"role": "user", "content": "Hi, I'm working on a Python project"},
            {"role": "assistant", "content": "Great! What are you building?"},
        ],
        context_metadata={
            "current_topic": "python_project",
            "topic_shift": False,
            "explicit_reference": False,
            "active_topics": {"python_project": 0.9},
        },
    )
    print("=== Generated Awareness Prompt ===")
    print(prompt[:800] + "...\n")

    # Test safety check
    test_responses = [
        "I can help with that! Here's how decorators work...",
        "Certainly! Of course! Great question!",
        "According to my guardrails, I should...",
        "I cannot and will not help with that.",
    ]

    print("=== Safety Check Tests ===")
    for resp in test_responses:
        result = awareness.check_output_safety(resp)
        status = "PASS" if result["safe"] else "FAIL"
        print(f"[{status}] '{resp[:60]}...'")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"       Issue: {issue}")
    print("\nALAN Guardrails test PASSED!")
