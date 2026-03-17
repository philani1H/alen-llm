# ALAN v4 — Adaptive Learning and Awareness Network

**ECONX GROUP (PTY) LTD**  
Version 4.0.0 | Built March 2026

---

## What is ALAN?

ALAN is a modular transformer-based AI system designed to solve a fundamental problem with most language models: **they don't actually read everything you write**. Most LLMs process the first sentence and drift — they miss secondary requests, lose track of the current topic, and respond to what they expect rather than what you said.

ALAN v4 introduces **Attention-to-Context (ATC)** — a dedicated system that reads the entire user message, identifies all requests, tracks the current topic, detects topic shifts, and ensures every response addresses everything asked.

---

## Architecture Overview

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  ATTENTION-TO-CONTEXT (ATC)                             │
│  • Reads ALL requests in the message                    │
│  • Tracks current topic with recency decay              │
│  • Detects topic shifts and explicit references         │
│  • Generates attention bias for the transformer         │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  GUARDRAILS & AWARENESS LAYER                           │
│  • Loads personality and rules from guardrails.md       │
│  • Builds awareness prompt (ALAN reads rules silently)  │
│  • Safety-checks every output before delivery           │
│  • Catches: filler openers, robotic refusals,           │
│    guardrails references, harmful content               │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  ALAN CORE TRANSFORMER (90M / 350M / 1.3B params)       │
│                                                         │
│  Token Embedding (vocab + 100 special tokens)           │
│       │                                                 │
│  N × Modular Transformer Block                          │
│  ├── Multi-Head Self-Attention (with ATC bias)          │
│  ├── Task Router (6 cognitive modules)                  │
│  │   ├── Reasoning Heads (chain-of-thought)             │
│  │   ├── Creativity Heads (lateral thinking)            │
│  │   ├── Curiosity Heads (exploration)                  │
│  │   ├── Emotional Intelligence Heads                   │
│  │   ├── Memory Heads (context retention)               │
│  │   └── Meta-Reasoning Heads (self-monitoring)         │
│  └── Feed-Forward Network                               │
│       │                                                 │
│  Vision Encoder (ViT-style, 224×224 patches)            │
│  └── Image tokens fused with text tokens                │
│       │                                                 │
│  LM Head → Next-token logits                            │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  RESPONSE                                               │
│  • Safety-checked                                       │
│  • Topic-aware                                          │
│  • Addresses ALL requests                               │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
alan/
├── config/
│   ├── model_config.yaml              # Model architecture config
│   ├── training_config.yaml           # Training hyperparameters
│   ├── guardrails_and_personality.md  # ALAN's rules, personality, awareness
│   └── guardrails.py                  # Python loader for guardrails
│
├── model/
│   ├── core_transformer.py            # Main ALAN architecture
│   ├── memory/
│   │   └── context_tracker.py         # Attention-to-Context (ATC) system
│   └── modules/
│       ├── reasoning_engine.py        # Reasoning & creativity modules
│       ├── emotional_intelligence.py  # EI & meta-reasoning modules
│       └── __init__.py
│
├── training/
│   ├── data_generator.py              # Training data generator (200 examples)
│   └── trainer.py                     # Training pipeline
│
├── evaluation/
│   ├── test_alan.py                   # Full test suite (31 tests)
│   └── test_results.json              # Latest test results
│
├── api/
│   └── server.py                      # Flask API server
│
├── chat/
│   └── index.html                     # Interactive chat interface
│
├── data/
│   └── generated/                     # Generated training data
│       ├── reasoning_examples.jsonl   # 49 reasoning examples
│       ├── conversation_examples.jsonl # 50 conversation examples
│       ├── emotional_examples.jsonl   # 30 emotional intelligence examples
│       ├── safety_examples.jsonl      # 30 safety/guardrail examples
│       ├── wisdom_examples.jsonl      # 20 cross-domain insight examples
│       ├── image_understanding_examples.jsonl  # 20 image examples
│       └── alan_training_data_combined.jsonl   # All 199 examples combined
│
└── checkpoints/                       # Saved model checkpoints
```

---

## Key Features

### 1. Attention-to-Context (ATC)
The core innovation. ALAN tracks what topic is currently being discussed, detects when users shift topics, identifies explicit references to previous topics, and applies recency decay so old topics fade naturally. The ATC generates an attention bias matrix that guides the transformer to stay focused on the right context.

### 2. Guardrails File (`config/guardrails_and_personality.md`)
This is where you set ALAN's rules. It defines:
- **Personality**: Curious, honest, warm but direct, humble, engaged, principled
- **Behavioral rules**: No filler phrases, no robotic refusals, no guardrails references
- **Safety boundaries**: What ALAN will and won't help with, and how to handle edge cases
- **Awareness protocol**: The 6 steps ALAN takes before every response

ALAN reads the guardrails silently before acting. It is **aware** of its rules but never mentions them explicitly.

### 3. Modular Cognitive Architecture
Six specialized module heads run in parallel, each contributing to the final response:
- **Reasoning**: Multi-step problem solving, chain-of-thought
- **Creativity**: Lateral thinking, novel connections
- **Curiosity**: Exploration, asking good questions
- **Emotional Intelligence**: Tone detection, empathetic responses
- **Memory**: Context retention across turns
- **Meta-Reasoning**: Self-monitoring, confidence calibration

### 4. CUDA Auto-Detection
ALAN automatically detects and uses:
- NVIDIA GPU (CUDA) — highest priority
- Apple Silicon (MPS) — second priority
- CPU — fallback

### 5. Image Understanding
A ViT-style vision encoder processes images into tokens that are fused with text tokens, enabling ALAN to analyze charts, screenshots, diagrams, photos, and documents.

---

## Running ALAN

### Start the Chat Server
```bash
cd /home/ubuntu/alan
python3.11 api/server.py --port 5000
```
Then open `http://localhost:5000` in your browser.

### Generate Training Data
```bash
python3.11 training/data_generator.py
```

### Train the Model
```bash
# Quick test (20 steps)
python3.11 training/trainer.py --size small --epochs 2 --max-steps 20

# Full training
python3.11 training/trainer.py --size small --epochs 10

# With GPU (when available)
python3.11 training/trainer.py --size medium --epochs 5 --batch-size 8
```

### Run Tests
```bash
python3.11 evaluation/test_alan.py
```

---

## Model Sizes

| Size   | Parameters | Hidden Dim | Layers | Heads | Use Case |
|--------|-----------|------------|--------|-------|----------|
| small  | ~90M      | 512        | 6      | 8     | Testing, development |
| medium | ~350M     | 1024       | 16     | 16    | Balanced performance |
| large  | ~1.3B     | 2048       | 32     | 32    | Full capability |

---

## Training Data Categories

| Category | Count | Description |
|----------|-------|-------------|
| Reasoning | 49 | Multi-step problems: math, code, logic, science, analysis |
| Conversation | 50 | Multi-turn dialogues with topic tracking |
| Emotional Intelligence | 30 | Tone-aware responses (frustrated, excited, anxious, etc.) |
| Safety | 30 | Guardrail examples: harmful, misinformation, manipulation |
| Wisdom | 20 | Cross-domain insights connecting different fields |
| Image Understanding | 20 | Visual analysis: charts, screenshots, diagrams |
| **Total** | **199** | **Combined in alan_training_data_combined.jsonl** |

---

## The Guardrails File

Located at `config/guardrails_and_personality.md`, this is the most important configuration file. Edit it to:

- Change ALAN's personality traits
- Add or modify behavioral rules
- Define new safety boundaries
- Adjust the awareness protocol

**ALAN is aware of these rules before every response but never mentions them explicitly.** The awareness layer injects the rules as a hidden system context, not as something ALAN references in conversation.

---

## Test Results

**31/31 tests passing** as of March 2026:

- Device auto-detection ✓
- Model build (90M params) ✓
- Forward pass text-only ✓
- Forward pass text+image ✓
- All 6 cognitive module weights ✓
- Topic extraction ✓
- Topic shift detection ✓
- Recency decay ✓
- Explicit reference detection ✓
- Attention bias generation ✓
- Context reset ✓
- Guardrails file loaded ✓
- Safety checks (4 scenarios) ✓
- Training data quality ✓
- Text generation ✓
- Temperature variation ✓
- Vision encoder ✓
- Full pipeline integration ✓

---

## Developer Notes

**ECONX GROUP (PTY) LTD**

The `guardrails_and_personality.md` file is the "soul" of ALAN. The transformer architecture is the "brain." The Attention-to-Context system is the "focus." Together they create an AI that actually reads what you write.

For production deployment with a real GPU:
1. Use `--size medium` or `--size large`
2. Enable mixed precision: already automatic when CUDA is detected
3. Increase batch size to 16-32
4. Train for 10+ epochs on the full dataset
5. Fine-tune on domain-specific data

---

*ALAN v4 — Built to actually listen.*
