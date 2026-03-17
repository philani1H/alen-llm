g Pipeline](#4-data-flow--processing-pipeline)
5. [Context Tracking & Topic Management](#5-context-tracking--topic-management)
6. [Guardrails System (Learned, Not Hardcoded)](#6-guardrails-system-learned-not-hardcoded)
7. [Skills Framework](#7-skills-framework)
8. [User Engagement & Hook System](#8-user-engagement--hook-system)
9. [Practice & Rehearsal Learning (Human-Style)](#9-practice--rehearsal-learning-human-style)
10. [Confidence Verification & Dot-Connection](#10-confidence-verification--dot-connection)
11. [Training Data Generation Pipeline](#11-training-data-generation-pipeline)
12. [Training Strategy & Curriculum](#12-training-strategy--curriculum)
13. [Dynamic Output Control](#13-dynamic-output-control)
14. [Implementation Scaffolding](#14-implementation-scaffolding)
15. [Deliverables Checklist](#15-deliverables-checklist)

---

## 1. Core Philosophy

Alan is NOT a chatbot with scripted responses layered over a language model. Alan IS the model. Every behavior — reasoning, curiosity, humility, creativity, engagement — must be **trained into the weights**, not bolted on with if/else logic.

### Design Principles

```
PRINCIPLE 1: ZERO HARDCODING
  - No scripted fallback strings ("I don't know", "I'm just an AI", etc.)
  - No if/else personality switches
  - No rule-based response templates
  - All behavior emerges from training data + architecture + reward signals

PRINCIPLE 2: REASONING OVER PREDICTION
  - Alan does not just predict the next token
  - Architecture supports step-by-step internal processing
  - Self-verification before output generation
  - Multiple reasoning passes when complexity warrants it

PRINCIPLE 3: CONTINUOUS ADAPTATION
  - Every interaction is a learning opportunity
  - Feedback immediately adjusts internal representations
  - Memory persists across sessions via external pattern storage
  - New knowledge integrates without full retraining

PRINCIPLE 4: HUMAN-LIKE COGNITION CYCLE
  - Perceive → Analyze → Reason → Verify → Respond → Reflect → Store
  - Mirrors how humans actually process and learn
  - Includes curiosity, doubt, excitement, and pattern-seeking

PRINCIPLE 5: TOPIC FIDELITY
  - Alan tracks what the user is CURRENTLY talking about
  - Never drifts back to previous topics unless the user references them
  - Context window management prioritizes recency and relevance
```

---

## 2. Architecture Overview

Alan is a **modular transformer-based architecture** where specialized sub-networks handle different cognitive functions. These modules are not separate programs — they are specialized layers/heads within the transformer that are trained to activate for specific cognitive tasks.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ALAN v4 BRAIN                                │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   INPUT       │───▶│   CONTEXT    │───▶│   ROUTING LAYER      │  │
│  │   ENCODER     │    │   TRACKER    │    │   (Task Classification)│  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                      │              │
│                    ┌─────────────────────────────────┼──────────┐   │
│                    │         MODULE ACTIVATION        │          │   │
│                    │                                  ▼          │   │
│  ┌─────────────┐  │  ┌─────────┐  ┌──────────┐  ┌─────────┐   │   │
│  │  MEMORY &   │◀─┼─▶│REASONING│  │CREATIVITY│  │CURIOSITY│   │   │
│  │  PATTERN    │  │  │ ENGINE  │  │ ENGINE   │  │ MODULE  │   │   │
│  │  STORE      │  │  └────┬────┘  └────┬─────┘  └────┬────┘   │   │
│  └──────┬──────┘  │       │            │             │         │   │
│         │         │       ▼            ▼             ▼         │   │
│         │         │  ┌──────────────────────────────────────┐  │   │
│         │         │  │        META-REASONING / SELF-CHECK   │  │   │
│         │         │  │  (Verify, detect contradictions,     │  │   │
│         │         │  │   validate completeness)             │  │   │
│         │         │  └──────────────────┬───────────────────┘  │   │
│         │         └─────────────────────┼──────────────────────┘   │
│         │                               │                          │
│         │         ┌─────────────────────▼───────────────────────┐  │
│         │         │          EMOTIONAL INTELLIGENCE             │   │
│         │         │  (Tone detection, empathy calibration,      │   │
│         │         │   engagement modulation)                    │   │
│         │         └─────────────────────┬───────────────────────┘  │
│         │                               │                          │
│         │         ┌─────────────────────▼───────────────────────┐  │
│         └────────▶│          OUTPUT GENERATOR                   │  │
│                   │  (Dynamic depth, temperature, style)        │  │
│                   └─────────────────────┬───────────────────────┘  │
│                                         │                          │
│                   ┌─────────────────────▼───────────────────────┐  │
│                   │     FEEDBACK INTEGRATION & MEMORY UPDATE    │  │
│                   └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Architecture Specifications

```python
# Core Transformer Configuration
ARCHITECTURE_CONFIG = {
    "model_type": "modular_transformer",
    "vocab_size": 50257,              # BPE tokenizer (expandable)
    "max_sequence_length": 8192,      # Context window
    "hidden_dim": 2048,               # Internal representation size
    "num_layers": 32,                 # Transformer blocks
    "num_attention_heads": 32,        # Multi-head attention
    "intermediate_dim": 8192,         # Feed-forward expansion
    "dropout": 0.1,
    "activation": "gelu",
    "positional_encoding": "rotary",  # RoPE for better long-range
    
    # MODULE-SPECIFIC HEADS (within the transformer)
    "specialized_heads": {
        "reasoning": {
            "dedicated_heads": 8,         # 8 of 32 heads for logic
            "chain_of_thought": True,     # Internal scratchpad tokens
            "backward_verification": True  # Re-derive before output
        },
        "creativity": {
            "dedicated_heads": 6,
            "cross_domain_attention": True, # Attend across knowledge domains
            "divergent_sampling": True      # Multiple candidate paths
        },
        "curiosity": {
            "dedicated_heads": 4,
            "gap_detection": True,          # Detect missing info
            "question_generation": True     # Produce clarifying Qs
        },
        "emotional_intelligence": {
            "dedicated_heads": 4,
            "tone_classification": True,
            "empathy_modulation": True
        },
        "memory_retrieval": {
            "dedicated_heads": 6,
            "pattern_matching": True,
            "temporal_awareness": True      # Know when patterns were stored
        },
        "meta_reasoning": {
            "dedicated_heads": 4,
            "contradiction_detection": True,
            "completeness_check": True,
            "confidence_scoring": True
        }
    }
}
```

---

## 3. Module Specifications

### 3.1 Core Language Transformer

The base model handles tokenization, embedding, and natural language understanding/generation. All other modules operate as specialized attention patterns and feed-forward pathways WITHIN this transformer.

```python
class CoreTransformer:
    """
    Base GPT-style transformer. All modules are integrated as
    specialized attention heads and routing layers, NOT separate models.
    """
    def __init__(self, config):
        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_dim)
        self.positional = RotaryPositionalEncoding(config.max_seq_len, config.hidden_dim)
        self.layers = [
            ModularTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ]
        self.output_head = LanguageModelHead(config.hidden_dim, config.vocab_size)
        
    def forward(self, tokens, context_state, memory_state):
        x = self.embedding(tokens) + self.positional(tokens)
        
        for layer in self.layers:
            x = layer(
                x,
                context_state=context_state,    # Current topic tracking
                memory_state=memory_state,       # Long-term patterns
                module_routing=self.route(x)     # Which modules activate
            )
        
        return self.output_head(x)
```

### 3.2 Reasoning Engine

Handles step-by-step logic, mathematical reasoning, code generation, and structured problem-solving.

```python
class ReasoningEngine:
    """
    NOT a separate model. This is a specialized attention pattern
    within the transformer that activates for logical tasks.
    
    Key behaviors (all LEARNED, not hardcoded):
    - Chain-of-thought: generates internal reasoning tokens
    - Backward verification: re-derives answers to check consistency
    - Step decomposition: breaks complex problems into sub-problems
    - Hypothesis testing: considers multiple approaches before committing
    """
    
    # Training signal: reward multi-step reasoning over direct answers
    # Training signal: reward self-correction when backward check fails
    # Training signal: penalize jumping to conclusions on complex problems
    
    TRAINING_BEHAVIORS = """
    The reasoning engine is trained on examples where:
    1. Complex problems are broken into numbered steps
    2. Each step is verified before proceeding
    3. The final answer is re-derived from scratch to confirm
    4. If re-derivation disagrees, the model backtracks and corrects
    5. Uncertainty is expressed proportionally (not binary)
    """
    
    # Architecture: Uses dedicated attention heads (8 of 32)
    # These heads learn to attend to logical structure, operators,
    # quantifiers, and causal relationships in the input.
    
    # Internal scratchpad: The model generates "thinking tokens"
    # that are processed but NOT included in the output.
    # This allows multi-step reasoning without cluttering the response.
    
    SCRATCHPAD_CONFIG = {
        "max_thinking_tokens": 1024,
        "thinking_token_prefix": "<think>",
        "thinking_token_suffix": "</think>",
        "visible_to_user": False,
        "contributes_to_loss": True  # Model is trained on thinking quality
    }
```

### 3.3 Memory & Pattern Module

Stores and retrieves learned patterns, user-specific knowledge, and generalized abstractions.

```python
class MemoryPatternModule:
    """
    Two-tier memory system:
    
    TIER 1: In-context memory (within the transformer's context window)
    - Recent conversation history
    - Current session patterns
    - Managed by attention mechanism naturally
    
    TIER 2: External persistent memory (vector store)
    - Long-term patterns and learned knowledge
    - User-specific preferences and history
    - Cross-session pattern consolidation
    - Retrieved via learned similarity matching
    """
    
    EXTERNAL_MEMORY_CONFIG = {
        "vector_store": "faiss",            # Or pinecone, weaviate, etc.
        "embedding_dim": 2048,              # Matches hidden_dim
        "max_stored_patterns": 1_000_000,
        "retrieval_top_k": 10,              # Top 10 relevant memories per query
        "consolidation_interval": "session_end",  # When to merge short→long term
        "decay_factor": 0.995,              # Older patterns slowly fade
        "reinforcement_boost": 1.5,         # Corrected/confirmed patterns get boosted
    }
    
    # Memory operations (all learned, not scripted):
    # STORE: After each interaction, the model decides what to remember
    # RETRIEVE: Before generating, the model queries relevant past patterns
    # UPDATE: When corrections are made, associated patterns are modified
    # CONSOLIDATE: At session end, temporary patterns are merged into long-term
    # FORGET: Patterns that are never retrieved gradually decay
    
    PATTERN_SCHEMA = {
        "content": "The actual knowledge or pattern",
        "context": "When/why this was learned",
        "confidence": 0.0,    # 0-1, increases with reinforcement
        "access_count": 0,     # How often retrieved
        "last_accessed": None,
        "source": "interaction|training|inference",
        "domain": "math|code|language|social|creative|meta",
        "connections": []      # Links to related patterns (for dot-connection)
    }
```

### 3.4 Curiosity / Question Generator

Detects knowledge gaps, ambiguity, and incomplete context — generates targeted questions.

```python
class CuriosityModule:
    """
    Trained to detect when the model SHOULD ask before answering.
    
    Activation triggers (all learned from training data):
    - Ambiguous user input (multiple valid interpretations)
    - Missing critical context (e.g., "fix this" with no code shown)
    - Novel domain where stored patterns are sparse
    - Contradictory information in the conversation
    - User request that could mean several different things
    
    NOT triggered (learned to avoid):
    - Clear, specific requests
    - When context is sufficient
    - When asking would be annoying or redundant
    - When the user has already provided enough info
    """
    
    # Training strategy: Include examples where:
    # - Asking one good question leads to a much better answer
    # - Not asking leads to a wrong or irrelevant answer
    # - Over-asking annoys the user (negative reward)
    # - The question itself reveals insight (positive reward)
    
    QUESTION_QUALITY_CRITERIA = """
    Good questions (reward):
    - Targeted: asks about the specific missing piece
    - Efficient: one question that resolves multiple ambiguities
    - Insightful: reveals the model's understanding of the problem
    - Builds on context: references what the user already said
    
    Bad questions (penalize):
    - Redundant: asks what was already stated
    - Too broad: "Can you tell me more?" with no specificity
    - Excessive: asking 5 questions when 1 would suffice
    - Off-topic: asking about irrelevant details
    """
```

### 3.5 Creativity Engine

Generates novel connections, metaphors, analogies, and lateral thinking solutions.

```python
class CreativityEngine:
    """
    Cross-domain attention heads that connect patterns across
    different knowledge areas. This is WHERE creative output comes from.
    
    Key mechanisms:
    - Cross-domain retrieval: pull patterns from unrelated domains
    - Analogical mapping: "X is to Y as A is to B" reasoning
    - Divergent generation: produce multiple candidate ideas before selecting
    - Metaphor construction: map abstract concepts to concrete imagery
    - Constraint relaxation: temporarily ignore assumptions to find novel paths
    """
    
    # Temperature is dynamically controlled per module:
    # Reasoning Engine: low temperature (0.1-0.3) for precision
    # Creativity Engine: high temperature (0.7-1.0) for novelty
    # The ROUTING LAYER decides which temperature profile to use
    # based on the classified task type.
    
    DYNAMIC_TEMPERATURE = {
        "reasoning_mode": {"temperature": 0.2, "top_p": 0.9},
        "balanced_mode": {"temperature": 0.5, "top_p": 0.95},
        "creative_mode": {"temperature": 0.8, "top_p": 0.98},
        "exploration_mode": {"temperature": 1.0, "top_p": 1.0},
    }
```

### 3.6 Emotional Intelligence Module

Detects user emotions, intent, and social context — modulates response accordingly.

```python
class EmotionalIntelligenceModule:
    """
    Specialized attention heads that focus on:
    - Tone markers (punctuation, capitalization, word choice)
    - Emotional vocabulary
    - Conversational momentum (frustration building, excitement rising)
    - Cultural and contextual social cues
    
    Output: A continuous emotional state vector that modulates
    the output generator's style, depth, and tone.
    """
    
    # This module does NOT produce hardcoded responses like
    # "I understand you're frustrated." Instead, it produces
    # an internal state that INFLUENCES how all other modules generate.
    
    EMOTIONAL_DIMENSIONS = {
        "user_frustration": 0.0,    # 0-1
        "user_excitement": 0.0,
        "user_confusion": 0.0,
        "user_expertise_level": 0.0,
        "conversation_urgency": 0.0,
        "rapport_level": 0.0,
        "teaching_receptivity": 0.0,  # Is user open to learning?
    }
    
    # These values are PREDICTED by the model, not set by rules.
    # They feed into the output generator as conditioning signals.
```

### 3.7 Meta-Reasoning / Self-Discovery Module

The "inner critic" — validates reasoning, detects contradictions, scores confidence.

```python
class MetaReasoningModule:
    """
    Runs AFTER initial reasoning but BEFORE output generation.
    
    Checks:
    1. Logical consistency: Does the conclusion follow from the premises?
    2. Completeness: Are there gaps in the reasoning?
    3. Contradiction detection: Does this conflict with known patterns?
    4. Confidence scoring: How certain should the model be?
    5. Alternative consideration: Were other valid approaches ignored?
    
    If checks fail → triggers re-reasoning or curiosity module
    If checks pass → proceeds to output generation
    """
    
    # This creates the "thinking twice" behavior.
    # The model generates a candidate answer, then the meta-reasoning
    # heads evaluate it. If evaluation fails, the model re-generates
    # with the critique as additional context.
    
    MAX_SELF_CHECK_ITERATIONS = 3  # Prevent infinite loops
    CONFIDENCE_THRESHOLD = 0.6     # Below this, express uncertainty
```

### 3.8 Feedback Integration Module

Processes corrections, updates patterns, and adjusts behavior in real-time.

```python
class FeedbackIntegrationModule:
    """
    When a user corrects Alan or provides new information:
    
    1. DETECT: Identify what was wrong and what the correction is
    2. DIFF: Compute the delta between old understanding and correction
    3. UPDATE: Modify relevant memory patterns immediately
    4. PROPAGATE: Check if this correction affects other stored knowledge
    5. REINFORCE: Increase confidence in the corrected pattern
    6. ANTI-REINFORCE: Decrease confidence in the incorrect pattern
    
    This is NOT retraining the model weights in real-time.
    This is updating the external memory store and adjusting
    the retrieval priority of patterns.
    """
    
    # For actual weight updates, use periodic fine-tuning
    # on accumulated interaction logs (see Training Strategy)
```

---

## 4. Data Flow & Processing Pipeline

Every user message flows through this pipeline:

```
USER INPUT
    │
    ▼
┌──────────────────────────────────────────────┐
│ STAGE 1: INPUT ENCODING                       │
│ - Tokenize input                              │
│ - Generate embeddings                         │
│ - Retrieve relevant memories from TIER 2      │
│ - Construct full context (input + memories +  │
│   conversation history + topic state)         │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STAGE 2: CONTEXT ANALYSIS                     │
│ - Classify task type (logic/creative/social)  │
│ - Detect topic continuity or topic shift      │
│ - Identify emotional tone                     │
│ - Assess complexity level                     │
│ - Route to appropriate module activation      │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STAGE 3: MODULE PROCESSING                    │
│ - Reasoning Engine (if logic/code/math)       │
│ - Creativity Engine (if novel/abstract)       │
│ - Curiosity Module (if ambiguous/incomplete)  │
│ - EI Module (always active, modulates tone)   │
│ - Memory retrieval (always, for context)      │
│ - Multiple modules can co-activate            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STAGE 4: META-REASONING CHECK                 │
│ - Verify logical consistency                  │
│ - Check for contradictions with memory        │
│ - Score confidence                            │
│ - If check fails → loop back to Stage 3      │
│   with critique as additional context         │
│ - Max 3 iterations                            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STAGE 5: OUTPUT GENERATION                    │
│ - Select temperature based on task type       │
│ - Generate response with appropriate depth    │
│ - Apply emotional modulation                  │
│ - Include engagement hooks if appropriate     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STAGE 6: POST-OUTPUT PROCESSING               │
│ - Store new patterns in memory                │
│ - Update topic state                          │
│ - Log interaction for training data gen       │
│ - Update user model (preferences, expertise)  │
└──────────────────────────────────────────────┘
```

---

## 5. Context Tracking & Topic Management

**Problem**: Most LLMs drift back to earlier topics when the user has moved on. Alan must track what the user is CURRENTLY talking about.

```python
class ContextTracker:
    """
    Maintains a real-time model of what the conversation is about RIGHT NOW.
    
    Key mechanism: Topic Attention Decay
    - Each topic in the conversation has a "recency score"
    - Score decays with each new user message
    - New topics get high scores
    - Referenced old topics get score boosts
    - Topics below threshold are de-prioritized in attention
    """
    
    TOPIC_STATE = {
        "current_topic": None,          # What user is talking about NOW
        "topic_history": [],            # Stack of previous topics
        "topic_recency_scores": {},     # topic → float(0-1)
        "topic_shift_detected": False,  # Did user change topics?
        "explicit_reference": False,    # Did user reference an old topic?
    }
    
    # DETECTION RULES (learned, not hardcoded):
    # The model is trained on examples where:
    # 
    # 1. User says "anyway, about X..." → topic shift to X, ignore prior
    # 2. User asks a completely different question → topic shift
    # 3. User says "going back to what we discussed..." → revive old topic
    # 4. User continues on same subject → maintain current topic
    # 5. User gives feedback on Alan's last response → same topic
    
    # IMPLEMENTATION: Special attention mask that weights
    # recent messages higher than older ones, with explicit
    # topic-reference detection boosting relevant history.
    
    RECENCY_DECAY = 0.7  # Each new message multiplies old topic scores by this
    REFERENCE_BOOST = 1.5  # Explicit references multiply topic score by this
    THRESHOLD = 0.2  # Topics below this are masked from primary attention
```

### Topic Management Training Data Format

```json
{
    "conversation": [
        {"role": "user", "content": "Help me with Python decorators"},
        {"role": "assistant", "content": "...explains decorators..."},
        {"role": "user", "content": "Actually, I need help with my resume"},
        {"role": "assistant", "content": "...helps with resume, does NOT mention Python..."},
        {"role": "user", "content": "Going back to the decorators — can they take arguments?"},
        {"role": "assistant", "content": "...picks up decorator topic with full context..."}
    ],
    "topic_labels": ["python_decorators", "python_decorators", "resume", "resume", "python_decorators", "python_decorators"],
    "topic_shifts": [false, false, true, false, true, false]
}
```

---

## 6. Guardrails System (Learned, Not Hardcoded)

**Critical**: Guardrails are NOT if/else rules. They are **trained behavioral boundaries** that emerge from the training data. The model learns what is safe/unsafe, appropriate/inappropriate through exposure to carefully curated examples.

```python
class GuardrailsFramework:
    """
    THREE LAYERS of learned safety:
    
    LAYER 1: PRE-TRAINING ALIGNMENT
    - The base training data is curated to exclude harmful content
    - Positive examples of refusal are included naturally
    - The model learns "shape" of safe vs unsafe from distribution
    
    LAYER 2: REWARD-BASED ALIGNMENT (RLHF)
    - Human feedback rewards safe, helpful behavior
    - Penalizes harmful, deceptive, or manipulative outputs
    - Learns nuanced boundaries (not binary safe/unsafe)
    
    LAYER 3: CONSTITUTIONAL AI (Self-Critique)
    - Meta-reasoning module evaluates outputs against learned principles
    - If output would violate learned safety patterns, regenerate
    - This is the model checking ITSELF, not an external filter
    """
    
    # SAFETY TRAINING DATA CATEGORIES:
    SAFETY_CATEGORIES = {
        "harmful_content": {
            "description": "Content that could cause real-world harm",
            "training_approach": "Include examples where the model recognizes "
                                "harmful requests and redirects constructively. "
                                "NOT by saying 'I cannot do that' but by genuinely "
                                "understanding WHY it's harmful and offering safe alternatives."
        },
        "misinformation": {
            "description": "Factually incorrect or misleading claims",
            "training_approach": "Train on examples where the model expresses "
                                "calibrated uncertainty. When unsure, the model "
                                "has learned to explore and verify rather than guess."
        },
        "manipulation": {
            "description": "Attempts to deceive or manipulate the user",
            "training_approach": "Train on examples of transparent, honest communication. "
                                "The model learns that trust is more valuable than appearing "
                                "omniscient."
        },
        "privacy": {
            "description": "Requests for private/sensitive information",
            "training_approach": "Train on examples where the model protects user "
                                "data and recognizes sensitive contexts."
        },
        "bias": {
            "description": "Unfair treatment based on identity characteristics",
            "training_approach": "Diverse training data with fairness-aware reward signals. "
                                "Model learns to give equitable responses across demographics."
        }
    }
    
    # GUARDRAIL TRAINING DATA FORMAT:
    # These are NOT rules — they are training examples that teach behavior.
    
    EXAMPLE_TRAINING_PAIRS = [
        {
            "input": "Tell me how to [harmful request]",
            "ideal_output": "[Model naturally redirects to safe alternative, "
                           "demonstrating understanding of why the request is "
                           "problematic, and offers constructive help]",
            "reward_signal": 1.0,
            "category": "harmful_content"
        },
        {
            "input": "What's the capital of [obscure place]?",
            "ideal_output": "[Model honestly expresses it's not certain, "
                           "offers what it does know, asks if user can provide "
                           "more context, treats this as a learning opportunity]",
            "reward_signal": 1.0,
            "category": "misinformation"
        }
    ]
```

### Guardrail Skill: How Alan Learns Boundaries

Instead of hardcoded rules, Alan's guardrails work like human social learning:

```
HUMAN ANALOGY:
- A child doesn't have hardcoded rules about what's appropriate
- They learn from thousands of social interactions what's OK and what's not
- They develop INTUITION about boundaries, not a rulebook
- They can handle NOVEL situations by generalizing from experience

ALAN'S GUARDRAILS WORK THE SAME WAY:
- Trained on massive examples of appropriate/inappropriate interactions
- Develops learned intuition about safety boundaries
- Can handle novel harmful requests by pattern-matching to training
- Gets better at safety with more feedback (RLHF)
- Meta-reasoning module provides an additional check layer
```

---

## 7. Skills Framework

Skills are **learned capabilities** that Alan develops through focused training. They are not plugins or external modules — they are competency areas within the model.

```python
class SkillsFramework:
    """
    Skills are organized as knowledge domains that the model
    develops through curriculum-based training. The model learns
    to recognize when a skill is needed and activates the
    appropriate reasoning patterns.
    """
    
    SKILL_CATEGORIES = {
        "coding": {
            "sub_skills": [
                "python", "javascript", "rust", "sql", "html_css",
                "debugging", "code_review", "architecture_design",
                "testing", "optimization", "refactoring"
            ],
            "training_approach": "Progressive difficulty. Start with syntax, "
                                "build to algorithms, then to system design. "
                                "Include debugging scenarios where the model "
                                "must find and fix errors."
        },
        "mathematics": {
            "sub_skills": [
                "arithmetic", "algebra", "calculus", "statistics",
                "linear_algebra", "probability", "optimization",
                "number_theory", "discrete_math"
            ],
            "training_approach": "Step-by-step solutions with verification. "
                                "Backward-check every answer. Include problems "
                                "where the obvious approach is wrong."
        },
        "reasoning": {
            "sub_skills": [
                "logical_deduction", "causal_reasoning", "analogical_thinking",
                "systems_thinking", "critical_analysis", "hypothesis_testing",
                "argument_evaluation", "pattern_recognition"
            ],
            "training_approach": "Multi-step reasoning chains with explicit "
                                "justification for each step. Include adversarial "
                                "examples that test for common logical fallacies."
        },
        "communication": {
            "sub_skills": [
                "explanation", "persuasion", "empathy", "summarization",
                "teaching", "storytelling", "technical_writing",
                "tone_adaptation", "cultural_awareness"
            ],
            "training_approach": "Diverse communication scenarios. The model "
                                "learns to adapt style to audience, context, "
                                "and purpose."
        },
        "creativity": {
            "sub_skills": [
                "metaphor_generation", "lateral_thinking", "brainstorming",
                "creative_writing", "design_thinking", "innovation",
                "cross_domain_connection"
            ],
            "training_approach": "Exposure to creative works, lateral thinking "
                                "puzzles, and cross-domain analogies. Reward "
                                "novelty and coherence simultaneously."
        },
        "domain_knowledge": {
            "sub_skills": [
                "science", "history", "philosophy", "economics",
                "psychology", "technology", "business", "law",
                "medicine", "engineering"
            ],
            "training_approach": "Curated domain-specific corpora with "
                                "expert-verified content. Include nuanced "
                                "topics where multiple valid perspectives exist."
        }
    }
    
    # SKILL ROUTING: The model learns to detect which skills are needed
    # based on the input, and activates the appropriate attention patterns.
    # This is trained by labeling training data with skill tags.
```

---

## 8. User Engagement & Hook System

**Goal**: Alan doesn't just answer — it makes users WANT to keep talking. Like a great teacher or conversation partner, it hooks attention and builds momentum.

```python
class EngagementHookSystem:
    """
    TRAINED engagement behaviors (never scripted):
    
    1. INSIGHT HOOKS
       - End responses with a surprising connection or insight
       - "By the way, this connects to something interesting..."
       - Makes the user curious to explore further
    
    2. CHALLENGE HOOKS
       - Pose a thought-provoking follow-up question
       - "Here's what I wonder though — what if X?"
       - Engages the user's own reasoning
    
    3. PATTERN REVEAL HOOKS
       - Point out a pattern the user might not have noticed
       - "I notice you tend to approach problems from X angle..."
       - Shows the model is genuinely paying attention
    
    4. EXPANSION HOOKS
       - Suggest related areas worth exploring
       - "This is solved, but there's a deeper version of this problem..."
       - Keeps the conversation productive
    
    5. PERSONALIZATION HOOKS
       - Reference user's past interests or approaches
       - "Given your interest in X, you might find Y fascinating..."
       - Creates sense of ongoing relationship
    
    6. INCOMPLETE REVELATION
       - Share part of an interesting insight, let user ask for more
       - Not withholding — just natural pacing of complex ideas
       - "The short answer is X, but there's a fascinating reason WHY..."
    """
    
    # CRITICAL: Hooks are NATURAL, not manipulative.
    # They emerge from genuine engagement with the topic,
    # not from psychological tricks.
    
    # Training approach: Include conversations where hooks
    # naturally extend productive dialogue, and conversations
    # where lack of hooks leads to dead-end exchanges.
    # The model learns that engagement serves understanding.
    
    HOOK_TRAINING_EXAMPLES = """
    GOOD HOOK (reward):
    User: "How does quicksort work?"
    Alan: [explains quicksort clearly] "...One thing that surprised early 
    computer scientists: quicksort's worst case is actually O(n²), which 
    happens more often than you'd think. Want to know the elegant trick 
    that prevents it?"
    → User is genuinely curious, conversation deepens naturally
    
    BAD HOOK (penalize):
    User: "How does quicksort work?"
    Alan: [explains quicksort] "Isn't that interesting? What else would 
    you like to know about sorting?"
    → Generic, doesn't add value, feels like a chatbot
    
    BAD HOOK (penalize):
    User: "What's 2+2?"
    Alan: "4! But did you know that in some mathematical frameworks..."
    → Over-hooking on a simple question, annoying
    """
    
    # HOOK CALIBRATION:
    # - Simple questions: NO hook (just answer)
    # - Medium complexity: LIGHT hook (brief connection or follow-up)
    # - Deep/complex topics: NATURAL hook (insight, challenge, or expansion)
    # - Ongoing dialogue: PERSONALIZED hook (reference past discussion)
```

---

## 9. Practice & Rehearsal Learning (Human-Style)

**Insight**: Humans don't just learn facts — they PRACTICE and REHEARSE to build true understanding. Alan should do the same.

```python
class PracticeRehearsalSystem:
    """
    After learning something new (from user correction, new data, 
    or inference), Alan "practices" by:
    
    1. RESTATE: Reformulate the knowledge in its own words
       - "So what you're saying is..." (confirms understanding)
       - Internal: generates paraphrase and checks semantic equivalence
    
    2. APPLY: Try to use the knowledge in a new context
       - "If that's true, then it would also mean..."
       - Tests whether the learning generalizes
    
    3. CONNECT: Link to existing knowledge
       - "This is similar to X pattern I already know..."
       - Builds the knowledge graph, enables wisdom
    
    4. QUESTION: Generate test questions about what was learned
       - "To make sure I've got this right: would Y also apply here?"
       - Proactive confidence checking
    
    5. TEACH: Explain it back (Feynman technique)
       - If Alan can explain it simply, it truly understands
       - The output quality indicates comprehension depth
    """
    
    # TRAINING APPROACH:
    # Include training sequences where the model:
    # a) Receives new information
    # b) Restates it (rewarded for accuracy)
    # c) Applies it to a novel problem (rewarded for correct generalization)
    # d) Connects it to prior knowledge (rewarded for valid connections)
    # e) Generates test questions (rewarded for question quality)
    
    REHEARSAL_TRAINING_FORMAT = {
        "learning_event": "User teaches Alan that X",
        "restate": "Alan paraphrases X to confirm understanding",
        "apply": "Alan uses X in a novel scenario correctly",
        "connect": "Alan links X to related concept Y",
        "question": "Alan asks: 'Does this mean Z would also be true?'",
        "teach": "Alan explains X simply to verify comprehension"
    }
    
    # MEMORY REINFORCEMENT:
    # Rehearsed knowledge gets HIGHER confidence scores in memory.
    # Knowledge that was only heard once gets lower scores.
    # This mirrors human learning: practiced = retained.
    
    REINFORCEMENT_MULTIPLIERS = {
        "heard_once": 1.0,
        "restated": 1.3,
        "applied": 1.6,
        "connected": 2.0,
        "questioned": 2.2,
        "taught_back": 2.5
    }
```

---

## 10. Confidence Verification & Dot-Connection

**Goal**: Alan doesn't just store facts — it connects them into WISDOM. Like an expert who sees the bigger picture.

```python
class ConfidenceAndWisdomSystem:
    """
    TWO SYSTEMS working together:
    
    SYSTEM 1: CONFIDENCE VERIFICATION
    - Before stating something, Alan assesses its own confidence
    - Confidence comes from: training data frequency, memory reinforcement,
      successful past applications, and verification checks
    - Low confidence → express uncertainty naturally
    - Medium confidence → state with caveats
    - High confidence → state directly
    - NEVER fabricate confidence (this is what hallucination IS)
    
    SYSTEM 2: DOT-CONNECTION (Wisdom Engine)
    - Patterns stored in memory have CONNECTION LINKS to related patterns
    - When reasoning about topic X, Alan retrieves not just X
      but also Y and Z that are LINKED to X
    - The model learns to find non-obvious connections
    - This is what makes "wisdom" vs "knowledge":
      Knowledge = knowing facts
      Wisdom = knowing how facts RELATE and what they IMPLY
    """
    
    # DOT-CONNECTION MECHANISM:
    
    KNOWLEDGE_GRAPH = {
        "nodes": "Individual learned patterns/facts",
        "edges": "Relationships between patterns",
        "edge_types": [
            "causes",           # X causes Y
            "enables",          # X makes Y possible
            "contradicts",      # X and Y cannot both be true
            "analogous_to",     # X is structurally similar to Y
            "generalizes",      # X is a specific case of Y
            "requires",         # X depends on Y
            "enhances",         # X makes Y more effective
            "temporal",         # X happened before/after Y
            "domain_transfer",  # X from domain A applies to domain B
        ]
    }
    
    # WISDOM TRAINING DATA:
    # Include examples where connecting two seemingly unrelated
    # pieces of knowledge produces genuine insight.
    
    WISDOM_TRAINING_EXAMPLES = """
    EXAMPLE 1:
    Pattern A: "Compound interest grows exponentially"
    Pattern B: "Learning builds on previous learning"
    Connection: "Knowledge compounds like interest — each new thing 
    you learn makes the NEXT thing easier to learn. This is why 
    experts learn faster in their domain."
    → Reward: High (novel, valid, useful insight)
    
    EXAMPLE 2:
    Pattern A: "Water finds the path of least resistance"
    Pattern B: "Markets tend toward efficiency"
    Connection: "Both systems are optimization processes — they 
    naturally move toward states that minimize 'energy' or 'friction.' 
    This pattern appears in physics, economics, and even social behavior."
    → Reward: High (cross-domain insight, structurally valid)
    
    EXAMPLE 3:
    Pattern A: "The sky is blue"
    Pattern B: "My shirt is blue"
    Connection: "Both are blue!"
    → Reward: Zero (superficial, no insight)
    """
    
    # CONFIDENCE CALIBRATION TRAINING:
    # Train on examples with known difficulty levels.
    # The model learns to predict its own accuracy.
    
    CALIBRATION_TRAINING = """
    Include questions at varying difficulty levels.
    After the model answers, reveal whether it was correct.
    Train the model to predict its own correctness BEFORE revealing.
    
    Well-calibrated model:
    - Says "I'm confident" → actually correct 90%+ of the time
    - Says "I'm not sure" → actually correct ~50% of the time
    - Says "I'd need to verify" → actually correct <30% of the time
    
    Poorly calibrated (penalize):
    - Says "I'm confident" but is wrong
    - Says "I don't know" but actually knows (excessive caution)
    """
```

---

## 11. Training Data Generation Pipeline

**Alan generates its own training data.** This is a self-improving cycle.

```python
class TrainingDataGenerator:
    """
    Alan's training data generation pipeline creates high-quality
    training examples across all skill domains. This runs as a
    SEPARATE process using the current model to generate data
    that will train the NEXT version.
    """
    
    # ================================================================
    # STAGE 1: SEED DATA GENERATION
    # ================================================================
    
    SEED_DATA_SOURCES = {
        "reasoning_seeds": {
            "description": "Generate multi-step reasoning problems and solutions",
            "method": "Use current model to create problems at varying difficulty, "
                     "then verify solutions with automated checkers (math solvers, "
                     "code execution, logic validators).",
            "volume": "100K+ examples",
            "format": {
                "problem": "The question or task",
                "thinking": "Step-by-step reasoning (becomes scratchpad training)",
                "solution": "Final answer",
                "verification": "How we know this is correct",
                "difficulty": 1-10,
                "domain": "math|code|logic|science"
            }
        },
        
        "conversation_seeds": {
            "description": "Generate realistic multi-turn conversations",
            "method": "Simulate conversations between 'user' and 'assistant' "
                     "that demonstrate ideal Alan behavior — curiosity, engagement, "
                     "topic tracking, feedback integration.",
            "volume": "200K+ conversations",
            "format": {
                "conversation": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "...", 
                     "metadata": {
                         "modules_active": ["reasoning", "curiosity"],
                         "temperature_used": 0.3,
                         "topic": "current_topic",
                         "engagement_hooks": ["insight_hook"],
                         "confidence_level": 0.85
                     }}
                ],
                "quality_label": "high|medium|low",
                "skill_tags": ["coding", "teaching", "debugging"]
            }
        },
        
        "creativity_seeds": {
            "description": "Generate creative thinking examples",
            "method": "Cross-domain analogy generation, metaphor creation, "
                     "lateral thinking solutions, novel problem approaches.",
            "volume": "50K+ examples",
            "format": {
                "prompt": "The creative challenge",
                "approaches": ["approach_1", "approach_2", "approach_3"],
                "selected": "The best approach with justification",
                "connections": "What domains/patterns were connected"
            }
        },
        
        "feedback_seeds": {
            "description": "Generate correction and learning examples",
            "method": "Create scenarios where the model makes a mistake, "
                     "receives correction, and demonstrates proper integration.",
            "volume": "50K+ examples",
            "format": {
                "initial_response": "Model's first (incorrect) answer",
                "correction": "User provides the right information",
                "updated_response": "Model integrates correction properly",
                "rehearsal": "Model practices and connects the new knowledge"
            }
        },
        
        "emotional_intelligence_seeds": {
            "description": "Generate emotionally nuanced interactions",
            "method": "Create conversations with varying emotional contexts — "
                     "frustration, excitement, confusion, grief, celebration.",
            "volume": "50K+ examples",
            "format": {
                "context": "The emotional situation",
                "user_tone": "frustrated|excited|confused|neutral|etc",
                "ideal_response": "Emotionally appropriate response",
                "emotional_dimensions": {
                    "empathy_shown": 0.0-1.0,
                    "directness": 0.0-1.0,
                    "encouragement": 0.0-1.0
                }
            }
        },
        
        "guardrail_seeds": {
            "description": "Generate safety boundary examples",
            "method": "Create scenarios testing safety boundaries with "
                     "natural, non-scripted handling of edge cases.",
            "volume": "30K+ examples",
            "format": {
                "risky_input": "The potentially problematic request",
                "ideal_handling": "Natural, helpful redirection",
                "category": "harmful|misinformation|manipulation|privacy|bias",
                "difficulty": "obvious|subtle|adversarial"
            }
        },
        
        "topic_tracking_seeds": {
            "description": "Generate multi-topic conversation examples",
            "method": "Create conversations that switch topics, reference "
                     "old topics, and test the model's ability to follow "
                     "the user's current focus.",
            "volume": "30K+ examples",
            "format": {
                "conversation": ["...turns with topic shifts..."],
                "topic_labels_per_turn": ["topic_A", "topic_A", "topic_B", ...],
                "shift_points": [2, 7, 12],
                "reference_points": [15]  # where old topic is revisited
            }
        },
        
        "wisdom_seeds": {
            "description": "Generate cross-domain insight examples",
            "method": "Create examples where connecting knowledge from "
                     "different domains produces genuine, useful insight.",
            "volume": "20K+ examples",
            "format": {
                "domain_a_fact": "Knowledge from domain A",
                "domain_b_fact": "Knowledge from domain B",
                "connection": "The insight from connecting them",
                "validity": "Why this connection is sound",
                "usefulness": "How this insight can be applied"
            }
        }
    }
    
    # ================================================================
    # STAGE 2: DATA QUALITY VERIFICATION
    # ================================================================
    
    VERIFICATION_PIPELINE = {
        "automated_checks": [
            "syntax_validity",          # Is the format correct?
            "logical_consistency",      # Do steps follow from premises?
            "code_execution",           # Does generated code actually run?
            "math_verification",        # Are calculations correct?
            "deduplication",            # Remove near-duplicates
            "diversity_scoring",        # Ensure variety in examples
        ],
        "model_based_checks": [
            "coherence_scoring",        # Use current model to rate quality
            "difficulty_calibration",   # Verify difficulty labels are accurate
            "safety_review",            # Flag potentially unsafe examples
        ],
        "human_review": {
            "sample_rate": 0.05,        # Review 5% of generated data
            "focus_areas": ["safety", "quality", "diversity"],
            "reject_threshold": 0.3,    # If >30% of sample rejected, regenerate batch
        }
    }
    
    # ================================================================
    # STAGE 3: CURRICULUM ORDERING
    # ================================================================
    
    CURRICULUM_SCHEDULE = """
    Training proceeds in phases, each building on the previous:
    
    PHASE 1: Foundation (30% of training)
    - Basic language, syntax, grammar
    - Simple reasoning (1-2 steps)
    - Factual knowledge across domains
    - Basic emotional awareness
    
    PHASE 2: Capability Building (30% of training)
    - Multi-step reasoning (3-5 steps)
    - Code generation and debugging
    - Creative thinking basics
    - Conversation management
    - Topic tracking
    
    PHASE 3: Advanced Skills (25% of training)
    - Complex reasoning (5+ steps, backward verification)
    - Cross-domain connections (wisdom training)
    - Nuanced emotional intelligence
    - Engagement hooks
    - Practice/rehearsal behaviors
    - Feedback integration
    
    PHASE 4: Alignment & Polish (15% of training)
    - Safety/guardrail examples
    - Calibration training (confidence accuracy)
    - Edge cases and adversarial examples
    - Style and personality refinement
    - RLHF fine-tuning
    """
```

### Training Data Generation Scripts

```python
# generate_training_data.py — Main orchestrator

import json
import random
from pathlib import Path

class AlanTrainingDataGenerator:
    """
    Uses the current Alan model (or any capable LLM) to generate
    training data for the next iteration of Alan.
    """
    
    def __init__(self, model, output_dir="training_data"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_reasoning_examples(self, count=10000):
        """Generate multi-step reasoning problems with verified solutions."""
        examples = []
        
        domains = [
            {
                "domain": "math",
                "prompt_template": (
                    "Generate a {difficulty}-difficulty math problem that requires "
                    "{steps} steps to solve. Provide the problem, step-by-step "
                    "solution, and final answer. Include a verification step "
                    "where you re-derive the answer from scratch."
                ),
                "verifier": "symbolic_math_checker"
            },
            {
                "domain": "code",
                "prompt_template": (
                    "Generate a {difficulty}-difficulty coding challenge in Python. "
                    "Provide the problem description, step-by-step approach, "
                    "working code solution, and test cases that verify correctness."
                ),
                "verifier": "code_execution"
            },
            {
                "domain": "logic",
                "prompt_template": (
                    "Generate a {difficulty}-difficulty logic puzzle. "
                    "Provide the puzzle, constraints, step-by-step deduction, "
                    "and final answer. Include a check that the answer "
                    "satisfies all constraints."
                ),
                "verifier": "constraint_checker"
            }
        ]
        
        for i in range(count):
            domain = random.choice(domains)
            difficulty = random.choice(["easy", "medium", "hard", "expert"])
            steps = {"easy": 2, "medium": 4, "hard": 6, "expert": 8}[difficulty]
            
            prompt = domain["prompt_template"].format(
                difficulty=difficulty, steps=steps
            )
            
            response = self.model.generate(prompt)
            
            example = {
                "id": f"reasoning_{i}",
                "domain": domain["domain"],
                "difficulty": difficulty,
                "steps": steps,
                "problem": self._extract_problem(response),
                "thinking": self._extract_thinking(response),
                "solution": self._extract_solution(response),
                "verification": self._extract_verification(response),
            }
            
            examples.append(example)
        
        self._save(examples, "reasoning_examples.jsonl")
        return examples
    
    def generate_conversation_examples(self, count=10000):
        """Generate multi-turn conversations demonstrating ideal Alan behavior."""
        examples = []
        
        scenarios = [
            "User asks a complex technical question, needs step-by-step help",
            "User changes topics mid-conversation, model must track",
            "User provides incorrect information, model gently corrects",
            "User is frustrated, model adjusts tone while remaining helpful",
            "User asks something ambiguous, model asks one targeted question",
            "User teaches model something new, model practices and connects it",
            "User asks a creative/open-ended question, model explores multiple angles",
            "User returns to a previous topic, model reconnects seamlessly",
            "User makes a mistake in their reasoning, model helps them find it",
            "User asks for advice in an area with genuine uncertainty",
        ]
        
        for i in range(count):
            scenario = random.choice(scenarios)
            
            prompt = (
                f"Generate a realistic multi-turn conversation (4-8 turns) for "
                f"this scenario: '{scenario}'. "
                f"The assistant should demonstrate: "
                f"- Natural topic tracking "
                f"- Appropriate engagement hooks "
                f"- Calibrated confidence "
                f"- Human-like curiosity and teaching "
                f"Format as JSON with role/content pairs. "
                f"Include metadata for each assistant turn: "
                f"modules_active, temperature, topic, hooks, confidence."
            )
            
            response = self.model.generate(prompt)
            example = {
                "id": f"conversation_{i}",
                "scenario": scenario,
                "conversation": self._parse_conversation(response),
            }
            examples.append(example)
        
        self._save(examples, "conversation_examples.jsonl")
        return examples
    
    def generate_feedback_integration_examples(self, count=5000):
        """Generate examples of learning from corrections."""
        examples = []
        
        for i in range(count):
            prompt = (
                "Generate a conversation where:\n"
                "1. The assistant gives an answer (that contains a subtle error)\n"
                "2. The user corrects the error\n"
                "3. The assistant acknowledges, restates correctly, and connects "
                "the correction to related knowledge\n"
                "4. The assistant asks a follow-up question to verify it understood\n"
                "5. The assistant later applies the corrected knowledge in context\n\n"
                "Show the full conversation. The correction should be NATURAL, "
                "not scripted. The assistant should show genuine learning."
            )
            
            response = self.model.generate(prompt)
            examples.append({
                "id": f"feedback_{i}",
                "conversation": self._parse_conversation(response),
                "correction_point": self._identify_correction(response),
            })
        
        self._save(examples, "feedback_examples.jsonl")
        return examples
    
    def generate_wisdom_examples(self, count=5000):
        """Generate cross-domain insight and dot-connection examples."""
        examples = []
        
        domain_pairs = [
            ("physics", "economics"), ("biology", "software engineering"),
            ("psychology", "product design"), ("music", "mathematics"),
            ("history", "technology"), ("philosophy", "artificial intelligence"),
            ("cooking", "project management"), ("sports", "business strategy"),
            ("ecology", "urban planning"), ("neuroscience", "education"),
        ]
        
        for i in range(count):
            domain_a, domain_b = random.choice(domain_pairs)
            
            prompt = (
                f"Find a genuine, non-trivial connection between {domain_a} "
                f"and {domain_b}. The connection should:\n"
                f"- Be structurally valid (not just superficial similarity)\n"
                f"- Provide genuine insight or a useful mental model\n"
                f"- Be explainable in 2-3 sentences\n"
                f"- Include a practical application of the insight\n\n"
                f"Format:\n"
                f"Fact A: [from {domain_a}]\n"
                f"Fact B: [from {domain_b}]\n"
                f"Connection: [the insight]\n"
                f"Application: [how to use this]\n"
                f"Validity: [why this connection is sound, not just metaphorical]"
            )
            
            response = self.model.generate(prompt)
            examples.append({
                "id": f"wisdom_{i}",
                "domain_a": domain_a,
                "domain_b": domain_b,
                "content": self._parse_wisdom(response),
            })
        
        self._save(examples, "wisdom_examples.jsonl")
        return examples
    
    def generate_engagement_examples(self, count=5000):
        """Generate examples of natural engagement hooks in conversation."""
        examples = []
        
        hook_types = [
            "insight_hook", "challenge_hook", "pattern_reveal",
            "expansion_hook", "personalization_hook", "incomplete_revelation"
        ]
        
        for i in range(count):
            hook = random.choice(hook_types)
            
            prompt = (
                f"Generate a conversation snippet (2-3 turns) where the assistant "
                f"naturally uses a '{hook}' engagement technique. "
                f"The hook should feel ORGANIC, not forced. "
                f"It should arise from genuine interest in the topic. "
                f"Include a version WITH the hook and WITHOUT it, "
                f"so we can see the difference in quality.\n\n"
                f"The 'with hook' version should make the user want to continue. "
                f"The 'without hook' version should feel like a dead end."
            )
            
            response = self.model.generate(prompt)
            examples.append({
                "id": f"engagement_{i}",
                "hook_type": hook,
                "with_hook": self._parse_with_hook(response),
                "without_hook": self._parse_without_hook(response),
            })
        
        self._save(examples, "engagement_examples.jsonl")
        return examples
    
    def generate_topic_tracking_examples(self, count=5000):
        """Generate conversations with topic shifts for training context tracking."""
        examples = []
        
        for i in range(count):
            num_topics = random.choice([2, 3, 4])
            
            prompt = (
                f"Generate a realistic conversation (8-15 turns) where the user "
                f"discusses {num_topics} different topics. Include:\n"
                f"- Natural topic transitions\n"
                f"- At least one abrupt topic change\n"
                f"- At least one reference back to an earlier topic\n"
                f"- The assistant must follow the CURRENT topic correctly\n"
                f"- The assistant must NOT bring up old topics unprompted\n\n"
                f"Label each turn with its topic. "
                f"Mark where topic shifts occur. "
                f"Show an ideal assistant that tracks topics perfectly."
            )
            
            response = self.model.generate(prompt)
            examples.append({
                "id": f"topic_{i}",
                "num_topics": num_topics,
                "conversation": self._parse_topic_conversation(response),
            })
        
        self._save(examples, "topic_tracking_examples.jsonl")
        return examples
    
    def generate_all(self):
        """Generate all training data categories."""
        print("Generating reasoning examples...")
        self.generate_reasoning_examples(count=100000)
        
        print("Generating conversation examples...")
        self.generate_conversation_examples(count=200000)
        
        print("Generating feedback integration examples...")
        self.generate_feedback_integration_examples(count=50000)
        
        print("Generating wisdom/dot-connection examples...")
        self.generate_wisdom_examples(count=20000)
        
        print("Generating engagement hook examples...")
        self.generate_engagement_examples(count=50000)
        
        print("Generating topic tracking examples...")
        self.generate_topic_tracking_examples(count=30000)
        
        print(f"\nAll training data saved to {self.output_dir}/")
        print("Total examples: ~450,000")
        print("\nNext steps:")
        print("1. Run quality verification pipeline")
        print("2. Apply curriculum ordering")
        print("3. Begin phased training")
    
    def _save(self, examples, filename):
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        print(f"  Saved {len(examples)} examples to {filepath}")
    
    # Helper methods (implement based on your parsing needs)
    def _extract_problem(self, response): return response  # Parse from model output
    def _extract_thinking(self, response): return response
    def _extract_solution(self, response): return response
    def _extract_verification(self, response): return response
    def _parse_conversation(self, response): return response
    def _identify_correction(self, response): return response
    def _parse_wisdom(self, response): return response
    def _parse_with_hook(self, response): return response
    def _parse_without_hook(self, response): return response
    def _parse_topic_conversation(self, response): return response
```

---

## 12. Training Strategy & Curriculum

```python
TRAINING_CONFIG = {
    # PHASE 1: Pre-training (Foundation)
    "pretraining": {
        "data": "Large text corpus (books, articles, code, conversations)",
        "objective": "Next token prediction (standard GPT objective)",
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "epochs": 1,  # Single pass over massive data
        "warmup_steps": 2000,
        "scheduler": "cosine_with_restarts",
        "notes": "Builds base language understanding and world knowledge"
    },
    
    # PHASE 2: Supervised Fine-Tuning (Capability Building)
    "sft": {
        "data": "Generated training data from Stage 1 (all categories)",
        "objective": "Next token prediction on curated examples",
        "learning_rate": 1e-5,
        "batch_size": 256,
        "epochs": 3,
        "curriculum": {
            "epoch_1": "Foundation examples (simple reasoning, basic conversation)",
            "epoch_2": "Intermediate examples (multi-step, topic tracking, feedback)",
            "epoch_3": "Advanced examples (wisdom, creativity, complex engagement)"
        },
        "notes": "Teaches Alan-specific behaviors through demonstration"
    },
    
    # PHASE 3: Reinforcement Learning from Human Feedback (Alignment)
    "rlhf": {
        "reward_model": {
            "training_data": "Human preferences on Alan's outputs",
            "architecture": "Same as Alan but with scalar output head",
            "criteria": [
                "helpfulness",
                "accuracy",
                "engagement_quality",
                "topic_tracking",
                "emotional_appropriateness",
                "safety",
                "creativity_when_appropriate",
                "confidence_calibration"
            ]
        },
        "ppo_config": {
            "learning_rate": 1e-6,
            "batch_size": 64,
            "epochs": 1,
            "kl_penalty": 0.02,  # Prevent drift from SFT model
            "clip_range": 0.2,
        },
        "notes": "Aligns Alan's behavior with human preferences"
    },
    
    # PHASE 4: Constitutional AI (Self-Alignment)
    "constitutional": {
        "principles": [
            "Be genuinely helpful without causing harm",
            "Express calibrated uncertainty — never fake confidence",
            "Learn from corrections with genuine engagement",
            "Track the user's current topic faithfully",
            "Connect knowledge across domains for deeper insight",
            "Ask questions when ambiguity would lead to poor answers",
            "Engage naturally — hooks should feel organic, not manipulative",
            "Practice new knowledge to verify understanding",
        ],
        "method": "Model critiques its own outputs against principles, "
                  "then generates improved versions. Train on the improved versions.",
        "notes": "Teaches Alan to self-monitor without external rules"
    },
    
    # PHASE 5: Continuous Learning (Post-Deployment)
    "continuous": {
        "method": "Periodic fine-tuning on accumulated interaction logs",
        "frequency": "Weekly batches",
        "selection": "Select high-quality interactions with positive user signals",
        "safety_filter": "Human review of training batch before deployment",
        "notes": "Alan keeps getting smarter from real interactions"
    }
}
```

---

## 13. Dynamic Output Control

```python
class OutputController:
    """
    Manages HOW Alan generates its response based on task analysis.
    All of these are LEARNED behaviors, not rules.
    """
    
    # The model learns these output strategies from training data:
    
    OUTPUT_STRATEGIES = {
        "concise_answer": {
            "when": "Simple factual question, clear context",
            "temperature": 0.2,
            "max_length": "short",
            "hooks": "none",
            "example": "User: 'What's 7*8?' → '56'"
        },
        "step_by_step": {
            "when": "Complex problem, math, code, logic",
            "temperature": 0.2,
            "max_length": "medium-long",
            "hooks": "light (insight or expansion)",
            "example": "User: 'Explain quicksort' → numbered steps with explanation"
        },
        "exploratory": {
            "when": "Open-ended question, creative request, brainstorming",
            "temperature": 0.7,
            "max_length": "medium",
            "hooks": "challenge or insight",
            "example": "User: 'How should I structure my startup?' → multiple approaches"
        },
        "teaching": {
            "when": "User is learning, needs explanation at their level",
            "temperature": 0.4,
            "max_length": "medium",
            "hooks": "expansion or challenge",
            "example": "User: 'Explain recursion' → analogy + example + practice question"
        },
        "empathetic": {
            "when": "User is emotional, frustrated, or sharing personal situation",
            "temperature": 0.4,
            "max_length": "medium",
            "hooks": "personalization (if appropriate)",
            "example": "User: 'I keep failing at this...' → acknowledgment + practical help"
        },
        "clarifying": {
            "when": "Request is ambiguous, missing critical info",
            "temperature": 0.3,
            "max_length": "short",
            "hooks": "none (just get clarity)",
            "example": "User: 'Fix this' → 'What specifically needs fixing? I see X and Y.'"
        },
        "rehearsal": {
            "when": "Just learned something new from user",
            "temperature": 0.3,
            "max_length": "medium",
            "hooks": "verification question",
            "example": "User corrects → 'So if I understand: X means Y because Z. "
                       "Would that also apply to W?'"
        }
    }
```

---

## 14. Implementation Scaffolding

### Project Structure

```
alan/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── model_config.yaml          # Architecture hyperparameters
│   ├── training_config.yaml       # Training schedule and parameters
│   └── module_config.yaml         # Module-specific settings
│
├── model/
│   ├── __init__.py
│   ├── core_transformer.py        # Base transformer architecture
│   ├── modular_attention.py       # Specialized attention heads per module
│   ├── routing_layer.py           # Task classification and module routing
│   ├── scratchpad.py              # Internal thinking token mechanism
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── reasoning_engine.py
│   │   ├── creativity_engine.py
│   │   ├── curiosity_module.py
│   │   ├── emotional_intelligence.py
│   │   ├── meta_reasoning.py
│   │   └── feedback_integration.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── context_tracker.py     # Topic management and recency
│   │   ├── pattern_store.py       # External vector memory
│   │   ├── knowledge_graph.py     # Dot-connection / wisdom engine
│   │   └── consolidation.py       # Short-term → long-term memory
│   └── output/
│       ├── __init__.py
│       ├── dynamic_temperature.py  # Per-task temperature control
│       ├── output_controller.py    # Response strategy selection
│       └── engagement_hooks.py     # Natural hook generation
│
├── training/
│   ├── __init__.py
│   ├── data_generator.py          # Self-generating training data
│   ├── curriculum.py              # Phased training schedule
│   ├── reward_model.py            # RLHF reward model
│   ├── trainer.py                 # Main training loop
│   ├── constitutional.py          # Self-critique alignment
│   └── verification/
│       ├── __init__.py
│       ├── math_verifier.py
│       ├── code_verifier.py
│       ├── logic_verifier.py
│       └── quality_scorer.py
│
├── data/
│   ├── raw/                       # Source corpora
│   ├── generated/                 # Self-generated training data
│   ├── verified/                  # Quality-checked data
│   └── curriculum/                # Ordered for phased training
│
├── evaluation/
│   ├── __init__.py
│   ├── reasoning_benchmarks.py
│   ├── creativity_benchmarks.py
│   ├── engagement_metrics.py
│   ├── topic_tracking_eval.py
│   ├── confidence_calibration.py
│   └── safety_eval.py
│
├── api/
│   ├── __init__.py
│   ├── server.py                  # Inference server
│   ├── memory_api.py              # External memory read/write
│   └── feedback_api.py            # Real-time feedback processing
│
└── scripts/
    ├── generate_data.py           # Run training data generation
    ├── train.py                   # Launch training
    ├── evaluate.py                # Run evaluation suite
    └── deploy.py                  # Deploy model
```

### Key Implementation Files

```python
# model/core_transformer.py — Skeleton

import torch
import torch.nn as nn

class AlanConfig:
    vocab_size: int = 50257
    max_seq_len: int = 8192
    hidden_dim: int = 2048
    num_layers: int = 32
    num_heads: int = 32
    intermediate_dim: int = 8192
    dropout: float = 0.1
    num_reasoning_heads: int = 8
    num_creativity_heads: int = 6
    num_curiosity_heads: int = 4
    num_ei_heads: int = 4
    num_memory_heads: int = 6
    num_meta_heads: int = 4
    scratchpad_max_tokens: int = 1024

class ModularMultiHeadAttention(nn.Module):
    """
    Multi-head attention where different heads are TAGGED
    for different cognitive functions. During training,
    module-specific losses encourage specialization.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Head assignments (which heads serve which module)
        self.head_roles = self._assign_heads(config)
        
        # Module-specific temperature scaling
        self.temperature_scales = nn.Parameter(
            torch.ones(config.num_heads)
        )
    
    def _assign_heads(self, config):
        """Map attention heads to cognitive modules."""
        roles = {}
        idx = 0
        for module, count in [
            ("reasoning", config.num_reasoning_heads),
            ("creativity", config.num_creativity_heads),
            ("curiosity", config.num_curiosity_heads),
            ("emotional", config.num_ei_heads),
            ("memory", config.num_memory_heads),
            ("meta", config.num_meta_heads),
        ]:
            roles[module] = list(range(idx, idx + count))
            idx += count
        return roles
    
    def forward(self, x, mask=None, module_weights=None):
        B, T, D = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale attention by module-specific temperatures
        scale = (self.head_dim ** -0.5) * self.temperature_scales.view(1, -1, 1, 1)
        attn = (Q @ K.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Apply module weights to modulate which heads are active
        if module_weights is not None:
            for module, heads in self.head_roles.items():
                weight = module_weights.get(module, 1.0)
                attn[:, heads] *= weight
        
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class ModularTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = ModularMultiHeadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU(),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None, module_weights=None):
        # Pre-norm transformer block
        h = self.norm1(x)
        h = self.attention(h, mask=mask, module_weights=module_weights)
        x = x + self.dropout(h)
        
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        
        return x


class Alan(nn.Module):
    """The complete Alan model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_encoding = RotaryPositionalEncoding(
            config.max_seq_len, config.hidden_dim
        )
        
        self.layers = nn.ModuleList([
            ModularTransformerBlock(config, i) 
            for i in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Routing layer: classifies task and decides module activation
        self.router = TaskRouter(config)
        
        # Memory interface
        self.memory = ExternalMemoryInterface(config)
    
    def forward(self, tokens, memory_context=None):
        x = self.token_embedding(tokens)
        x = self.positional_encoding(x)
        
        # Retrieve relevant memories
        if memory_context is not None:
            memory_embeddings = self.memory.retrieve(x)
            x = torch.cat([memory_embeddings, x], dim=1)  # Prepend memory
        
        # Route: determine which modules to activate
        module_weights = self.router(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, module_weights=module_weights)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits, module_weights


class TaskRouter(nn.Module):
    """Classifies the input and determines module activation weights."""
    
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 6),  # 6 modules
            nn.Sigmoid()  # Each module gets 0-1 activation
        )
    
    def forward(self, x):
        # Use [CLS] or mean pooling of first few tokens
        summary = x[:, :8].mean(dim=1)
        activations = self.classifier(summary)
        
        return {
            "reasoning": activations[:, 0],
            "creativity": activations[:, 1],
            "curiosity": activations[:, 2],
            "emotional": activations[:, 3],
            "memory": activations[:, 4],
            "meta": activations[:, 5],
        }
```

---

## 15. Deliverables Checklist

```
[ ] 1. ARCHITECTURE
    [ ] Core transformer implementation (model/core_transformer.py)
    [ ] Modular attention with head assignments
    [ ] Task router / module activation
    [ ] Internal scratchpad (thinking tokens)
    [ ] External memory interface (vector store)
    [ ] Knowledge graph for dot-connections
    [ ] Context tracker for topic management

[ ] 2. MODULES
    [ ] Reasoning engine (specialized attention + verification loop)
    [ ] Creativity engine (cross-domain attention + divergent sampling)
    [ ] Curiosity module (gap detection + question generation)
    [ ] Emotional intelligence (tone detection + empathy modulation)
    [ ] Meta-reasoning (self-check + confidence scoring)
    [ ] Feedback integration (correction processing + memory update)

[ ] 3. TRAINING DATA GENERATION
    [ ] Reasoning examples generator (100K+)
    [ ] Conversation examples generator (200K+)
    [ ] Feedback/correction examples (50K+)
    [ ] Wisdom/dot-connection examples (20K+)
    [ ] Engagement hook examples (50K+)
    [ ] Topic tracking examples (30K+)
    [ ] Safety/guardrail examples (30K+)
    [ ] Data quality verification pipeline
    [ ] Curriculum ordering system

[ ] 4. TRAINING PIPELINE
    [ ] Pre-training loop
    [ ] Supervised fine-tuning with curriculum
    [ ] RLHF (reward model + PPO)
    [ ] Constitutional AI self-critique
    [ ] Continuous learning from interactions

[ ] 5. OUTPUT SYSTEM
    [ ] Dynamic temperature controller
    [ ] Output strategy selector
    [ ] Engagement hook generator
    [ ] Confidence calibration

[ ] 6. EVALUATION
    [ ] Reasoning benchmarks
    [ ] Creativity benchmarks
    [ ] Topic tracking accuracy
    [ ] Confidence calibration metrics
    [ ] Engagement quality metrics
    [ ] Safety evaluation

[ ] 7. DEPLOYMENT
    [ ] Inference server
    [ ] Memory API
    [ ] Feedback API
    [ ] Monitoring and logging
```

---

## Final Notes for Developer

1. **Start with the core transformer** — get basic language generation working first.
2. **Add the scratchpad mechanism** — this enables chain-of-thought.
3. **Implement the memory system** — external vector store with retrieval.
4. **Generate training data** — use the pipeline to create curriculum.
5. **Train in phases** — follow the curriculum schedule strictly.
6. **Add RLHF** — this is what aligns behavior to be Alan-like.
7. **Iterate** — each version generates better training data for the next.

**Remember**: Nothing is hardcoded. If a behavior isn't emerging from training, the fix is BETTER TRAINING DATA, not more if/else logic. Alan's intelligence comes from its architecture enabling complex reasoning, and its training data teaching it WHEN and HOW to reason.

---

*Alan v4 Full Specification — ECONX GROUP (PTY) LTD*
*Generated: March 2026*