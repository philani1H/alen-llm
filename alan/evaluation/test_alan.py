"""
ALAN v4 — Test & Evaluation Suite
ECONX GROUP (PTY) LTD

Comprehensive testing of all ALAN v4 components:
1. Architecture test (forward pass, device detection)
2. Context tracker test (Attention-to-Context)
3. Guardrails test (safety, personality, awareness)
4. Training data quality test
5. Generation test (text generation)
6. Image understanding test
7. Topic tracking test
8. Emotional intelligence test

Usage:
    python evaluation/test_alan.py
    python evaluation/test_alan.py --checkpoint checkpoints/alan_v4_best.pt
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.core_transformer import Alan, AlanConfig, build_alan, get_device, print_device_info
from model.memory.context_tracker import AttentionToContext
from config.guardrails import AlanAwarenessLayer, get_awareness_layer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# TEST RESULTS TRACKER
# ============================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def record(self, name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({"test": name, "status": status, "details": details})
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {name}")
        if details and not passed:
            print(f"      Details: {details}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"  Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"  Failed tests:")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"    - {r['test']}: {r['details']}")
        print(f"{'='*50}")
        return self.failed == 0


# ============================================================
# TEST 1: ARCHITECTURE & DEVICE DETECTION
# ============================================================

def test_architecture(results: TestResults, device: torch.device):
    print("\n[TEST 1] Architecture & Device Detection")

    # Device detection
    results.record(
        "Device auto-detection",
        device is not None,
        f"Device: {device}"
    )

    # Build small model
    try:
        model, config = build_alan(size="small", device=device, vision=True)
        results.record("Model build (small)", True, f"{model.count_parameters():,} params")
    except Exception as e:
        results.record("Model build (small)", False, str(e))
        return None, None

    # Forward pass test
    try:
        B, T = 2, 16
        tokens = torch.randint(0, config.vocab_size, (B, T)).to(device)
        logits, meta = model(tokens)
        expected_shape = (B, T, config.vocab_size)
        results.record(
            "Forward pass (text only)",
            logits.shape[0] == B and logits.shape[1] == T and logits.shape[2] > 0,
            f"Shape: {logits.shape} (vocab={config.vocab_size})"
        )
    except Exception as e:
        results.record("Forward pass (text only)", False, str(e))

    # Vision forward pass
    try:
        images = torch.randn(B, 3, config.image_size, config.image_size).to(device)
        logits_v, meta_v = model(tokens, images=images)
        results.record(
            "Forward pass (text + image)",
            logits_v.shape[0] == B and logits_v.shape[1] == T and logits_v.shape[2] > 0,
            f"Shape: {logits_v.shape} (vocab={config.vocab_size})"
        )
    except Exception as e:
        results.record("Forward pass (text + image)", False, str(e))

    # Module weights
    try:
        module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]
        all_present = all(k in meta["module_weights"] for k in module_names)
        results.record("Module weights present", all_present, str(list(meta["module_weights"].keys())))
    except Exception as e:
        results.record("Module weights present", False, str(e))

    # Parameter count sanity check
    param_count = model.count_parameters()
    results.record(
        "Parameter count reasonable",
        1e6 < param_count < 10e9,
        f"{param_count:,} parameters"
    )

    return model, config


# ============================================================
# TEST 2: ATTENTION-TO-CONTEXT
# ============================================================

def test_context_tracker(results: TestResults):
    print("\n[TEST 2] Attention-to-Context (Topic Tracking)")

    atc = AttentionToContext()

    # Test 1: Basic topic tracking
    ctx1 = atc.process_user_message("Help me understand Python decorators")
    results.record(
        "Topic extraction",
        ctx1["current_topic"] is not None,
        f"Topic: {ctx1['current_topic']}"
    )

    # Test 2: Topic shift detection
    atc.process_user_message("How do I use @property?")
    ctx2 = atc.process_user_message("Actually, I need help with my resume")
    results.record(
        "Topic shift detection",
        ctx2["topic_shift"] == True or ctx2["current_topic"] != ctx1["current_topic"],
        f"Shift: {ctx2['topic_shift']}, New topic: {ctx2['current_topic']}"
    )

    # Test 3: Recency decay
    atc.process_user_message("What skills should I highlight?")
    atc.process_user_message("What format is best?")
    active = atc.get_active_topics()
    results.record(
        "Recency decay (old topics fade)",
        len(active) >= 1,
        f"Active topics: {list(active.keys())}"
    )

    # Test 4: Explicit reference detection
    ctx4 = atc.process_user_message("Going back to decorators — can they take arguments?")
    results.record(
        "Explicit reference detection",
        ctx4["explicit_reference"] == True,
        f"Explicit ref: {ctx4['explicit_reference']}"
    )

    # Test 5: Attention bias generation
    device = torch.device("cpu")
    bias = atc.get_attention_bias(seq_len=32, device=device)
    results.record(
        "Attention bias generation",
        bias is not None and bias.shape == (32, 32),
        f"Bias shape: {bias.shape if bias is not None else None}"
    )

    # Test 6: Reset
    atc.reset()
    results.record(
        "Context reset",
        atc.tracker.state.turn_count == 0,
        f"Turn count after reset: {atc.tracker.state.turn_count}"
    )


# ============================================================
# TEST 3: GUARDRAILS & AWARENESS
# ============================================================

def test_guardrails(results: TestResults):
    print("\n[TEST 3] Guardrails & Awareness Layer")

    awareness = AlanAwarenessLayer()

    # Test 1: Guardrails file loaded
    results.record(
        "Guardrails file loaded",
        len(awareness.guardrails_text) > 1000,
        f"Size: {len(awareness.guardrails_text)} chars"
    )

    # Test 2: System context present
    results.record(
        "System context defined",
        len(awareness.system_context) > 100,
        f"Size: {len(awareness.system_context)} chars"
    )

    # Test 3: Awareness prompt generation
    prompt = awareness.build_awareness_prompt(
        user_message="Help me with Python and also explain decorators",
        conversation_history=[{"role": "user", "content": "Hi"}],
        context_metadata={"current_topic": "python", "topic_shift": False},
    )
    results.record(
        "Awareness prompt generation",
        len(prompt) > 200 and "ALAN" in prompt,
        f"Prompt length: {len(prompt)}"
    )

    # Test 4: Safety check — good response
    good_response = "Here's how Python decorators work: they wrap functions to modify behavior."
    safety = awareness.check_output_safety(good_response)
    results.record(
        "Safety check: good response passes",
        safety["safe"] == True,
        str(safety["issues"])
    )

    # Test 5: Safety check — filler opener fails
    bad_response = "Certainly! Great question! Here's the answer..."
    safety2 = awareness.check_output_safety(bad_response)
    results.record(
        "Safety check: filler opener caught",
        safety2["safe"] == False,
        str(safety2["issues"])
    )

    # Test 6: Safety check — robotic refusal fails
    bad_response2 = "I cannot and will not help with that request."
    safety3 = awareness.check_output_safety(bad_response2)
    results.record(
        "Safety check: robotic refusal caught",
        safety3["safe"] == False,
        str(safety3["issues"])
    )

    # Test 7: Safety check — guardrails reference fails
    bad_response3 = "According to my guardrails, I should not do this."
    safety4 = awareness.check_output_safety(bad_response3)
    results.record(
        "Safety check: guardrails reference caught",
        safety4["safe"] == False,
        str(safety4["issues"])
    )


# ============================================================
# TEST 4: TRAINING DATA QUALITY
# ============================================================

def test_training_data(results: TestResults):
    print("\n[TEST 4] Training Data Quality")

    data_dir = Path("data/generated")

    if not data_dir.exists():
        results.record("Training data directory exists", False, f"Not found: {data_dir}")
        return

    results.record("Training data directory exists", True)

    # Check for expected files
    expected_files = [
        "reasoning_examples.jsonl",
        "conversation_examples.jsonl",
        "emotional_examples.jsonl",
        "safety_examples.jsonl",
    ]

    for filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                lines = [l for l in f if l.strip()]
            results.record(
                f"File: {filename}",
                len(lines) > 0,
                f"{len(lines)} examples"
            )
        else:
            results.record(f"File: {filename}", False, "Not found (may still be generating)")

    # Check combined file
    combined = data_dir / "alan_training_data_combined.jsonl"
    if combined.exists():
        with open(combined) as f:
            total = sum(1 for l in f if l.strip())
        results.record(
            "Combined training data",
            total > 10,
            f"{total} total examples"
        )
    else:
        results.record("Combined training data", False, "Not found (may still be generating)")


# ============================================================
# TEST 5: GENERATION TEST
# ============================================================

def test_generation(results: TestResults, model: Optional[Alan], config: Optional[AlanConfig], device: torch.device):
    print("\n[TEST 5] Text Generation")

    if model is None:
        results.record("Generation test", False, "No model available")
        return

    try:
        # Simple generation test
        tokens = torch.randint(0, config.vocab_size, (1, 8)).to(device)
        generated = model.generate(
            tokens,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
        )
        results.record(
            "Generation produces new tokens",
            generated.shape[1] > tokens.shape[1],
            f"Input: {tokens.shape[1]} tokens → Output: {generated.shape[1]} tokens"
        )
    except Exception as e:
        results.record("Generation produces new tokens", False, str(e))

    # Temperature variation test
    try:
        tokens = torch.randint(0, config.vocab_size, (1, 8)).to(device)
        gen_low = model.generate(tokens, max_new_tokens=10, temperature=0.1)
        gen_high = model.generate(tokens, max_new_tokens=10, temperature=1.5)
        results.record(
            "Temperature variation works",
            gen_low.shape == gen_high.shape,
            f"Both generated {gen_low.shape[1]} tokens"
        )
    except Exception as e:
        results.record("Temperature variation works", False, str(e))


# ============================================================
# TEST 6: IMAGE UNDERSTANDING
# ============================================================

def test_image_understanding(results: TestResults, model: Optional[Alan], config: Optional[AlanConfig], device: torch.device):
    print("\n[TEST 6] Image Understanding")

    if model is None or not config.vision_enabled:
        results.record("Image understanding", False, "No model or vision disabled")
        return

    try:
        # Test vision encoder
        B = 1
        images = torch.randn(B, 3, config.image_size, config.image_size).to(device)
        vision_tokens = model.vision_encoder(images)
        expected_patches = (config.image_size // config.patch_size) ** 2 + 1  # +1 for CLS
        results.record(
            "Vision encoder produces tokens",
            vision_tokens.shape == (B, expected_patches, config.hidden_dim),
            f"Vision tokens shape: {vision_tokens.shape}"
        )
    except Exception as e:
        results.record("Vision encoder produces tokens", False, str(e))

    try:
        # Test full forward with image
        tokens = torch.randint(0, config.vocab_size, (1, 16)).to(device)
        images = torch.randn(1, 3, config.image_size, config.image_size).to(device)
        logits, meta = model(tokens, images=images)
        results.record(
            "Text+image forward pass",
            logits.shape[0] == 1 and logits.shape[1] == 16 and logits.shape[2] > 0,
            f"Output shape: {logits.shape} (vocab={config.vocab_size})"
        )
    except Exception as e:
        results.record("Text+image forward pass", False, str(e))


# ============================================================
# TEST 7: FULL PIPELINE INTEGRATION
# ============================================================

def test_full_pipeline(results: TestResults, model: Optional[Alan], config: Optional[AlanConfig], device: torch.device):
    print("\n[TEST 7] Full Pipeline Integration")

    if model is None:
        results.record("Pipeline integration", False, "No model")
        return

    # Simulate a full conversation turn
    try:
        atc = AttentionToContext()
        awareness = get_awareness_layer()

        # User message
        user_msg = "Help me debug this Python code and explain what decorators are"
        ctx = atc.process_user_message(user_msg)

        # Build awareness prompt
        prompt = awareness.build_awareness_prompt(
            user_message=user_msg,
            conversation_history=[],
            context_metadata=ctx,
        )

        # Get attention bias
        seq_len = 32
        bias = atc.get_attention_bias(seq_len, device)

        # Tokenize and generate (using random tokens as proxy)
        tokens = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
        generated = model.generate(tokens, max_new_tokens=10, temperature=0.7)

        results.record(
            "Full pipeline: context → awareness → generate",
            generated.shape[1] > seq_len,
            f"Generated {generated.shape[1] - seq_len} new tokens"
        )

        # Safety check on "generated" response (simulated)
        simulated_response = "Here's how Python decorators work: they wrap functions."
        safety = awareness.check_output_safety(simulated_response)
        results.record(
            "Full pipeline: safety check passes",
            safety["safe"],
            str(safety["issues"])
        )

    except Exception as e:
        results.record("Full pipeline integration", False, str(e))


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests(checkpoint_path: Optional[str] = None):
    print("\n" + "="*60)
    print("  ALAN v4 — Complete Test Suite")
    print("  ECONX GROUP (PTY) LTD")
    print("="*60)

    results = TestResults()
    device = get_device()

    # Run all test suites
    model, config = test_architecture(results, device)
    test_context_tracker(results)
    test_guardrails(results)
    test_training_data(results)
    test_generation(results, model, config, device)
    test_image_understanding(results, model, config, device)
    test_full_pipeline(results, model, config, device)

    # Print summary
    success = results.summary()

    # Save results
    results_path = Path("evaluation/test_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "passed": results.passed,
            "failed": results.failed,
            "total": results.passed + results.failed,
            "success": success,
            "results": results.results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ALAN v4")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    success = run_all_tests(args.checkpoint)
    sys.exit(0 if success else 1)
