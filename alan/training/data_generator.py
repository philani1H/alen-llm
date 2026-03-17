"""
ALAN v4 — Training Data Generator
ECONX GROUP (PTY) LTD

Generates comprehensive training data for all ALAN behaviors:
- Reasoning examples (multi-step, chain-of-thought)
- Conversation examples (topic tracking, engagement)
- Emotional intelligence examples
- Guardrail/safety examples
- Wisdom/cross-domain connection examples
- Image understanding examples (text descriptions)
- Feedback/correction examples

Uses OpenAI API to generate high-quality training data.
All output is saved as JSONL and TXT files.
"""

import os
import json
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output directories
DATA_DIR = Path(__file__).parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI client
client = OpenAI()
MODEL = "gpt-4.1-mini"


def call_llm(prompt: str, system: str = "", temperature: float = 0.8, max_tokens: int = 1500) -> str:
    """Call the LLM with retry logic."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""


# ============================================================
# ALAN SYSTEM PROMPT FOR DATA GENERATION
# ============================================================

ALAN_GENERATION_SYSTEM = """You are generating training data for ALAN v4 — an AI assistant 
built by ECONX GROUP. ALAN's key traits:
- Reads ALL of the user's requests, not just the first sentence
- Stays on the current topic (never drifts to previous topics unprompted)
- Honest about uncertainty (explores rather than fabricates)
- Warm but direct — no filler phrases like "Certainly!" or "Great question!"
- Curious and engaged — asks good follow-up questions
- Addresses EVERYTHING the user asked, in order of importance

Generate realistic, high-quality training examples. Be specific and detailed."""


# ============================================================
# GENERATOR 1: REASONING EXAMPLES
# ============================================================

def generate_reasoning_examples(count: int = 50) -> List[Dict]:
    """Generate multi-step reasoning training examples."""
    examples = []

    domains = [
        ("math", "arithmetic, algebra, or word problem"),
        ("code", "Python debugging or algorithm"),
        ("logic", "deductive reasoning puzzle"),
        ("science", "physics or chemistry problem"),
        ("analysis", "data interpretation or argument analysis"),
    ]

    difficulties = ["easy", "medium", "hard"]

    for i in range(count):
        domain_name, domain_desc = random.choice(domains)
        difficulty = random.choice(difficulties)
        steps = {"easy": 2, "medium": 4, "hard": 6}[difficulty]

        prompt = f"""Generate a {difficulty} {domain_desc} that requires {steps} clear steps to solve.

Format your response as JSON:
{{
  "problem": "The problem statement",
  "thinking": ["Step 1: ...", "Step 2: ...", ...],
  "solution": "Final answer with explanation",
  "verification": "Re-check: verify the answer is correct",
  "hook": "One interesting insight or follow-up question"
}}

Make it realistic and educational. The thinking steps should show genuine reasoning."""

        response = call_llm(prompt, system=ALAN_GENERATION_SYSTEM, temperature=0.7)

        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                example = {
                    "id": f"reasoning_{domain_name}_{i:04d}",
                    "type": "reasoning",
                    "domain": domain_name,
                    "difficulty": difficulty,
                    **data
                }
                examples.append(example)
                logger.info(f"  Generated reasoning example {i+1}/{count}: {domain_name}/{difficulty}")
        except json.JSONDecodeError:
            # Fallback: store raw response
            examples.append({
                "id": f"reasoning_{domain_name}_{i:04d}",
                "type": "reasoning",
                "domain": domain_name,
                "difficulty": difficulty,
                "raw": response,
            })

    return examples


# ============================================================
# GENERATOR 2: CONVERSATION EXAMPLES (Topic Tracking)
# ============================================================

def generate_conversation_examples(count: int = 50) -> List[Dict]:
    """Generate multi-turn conversations demonstrating ALAN's topic tracking."""
    examples = []

    scenarios = [
        "User asks about Python, then switches to asking about their resume",
        "User asks about machine learning, then asks a completely unrelated cooking question",
        "User discusses a coding problem, then goes back to a topic from 3 turns ago",
        "User asks multiple questions in one message about different topics",
        "User is frustrated with a bug and needs both technical help and emotional support",
        "User teaches ALAN something new, ALAN practices and connects it to other knowledge",
        "User asks an ambiguous question that needs clarification before answering",
        "User asks for advice on a decision with genuine uncertainty involved",
        "User asks a creative/open-ended question requiring exploration of multiple angles",
        "User corrects ALAN's previous response and ALAN integrates the correction",
    ]

    for i in range(count):
        scenario = random.choice(scenarios)

        prompt = f"""Generate a realistic multi-turn conversation (4-8 turns) for this scenario:
"{scenario}"

The assistant (ALAN) should demonstrate:
- Reading ALL of what the user asks (not just the first sentence)
- Staying on the current topic unless the user shifts
- Natural curiosity and engagement
- Honest confidence calibration
- No filler phrases ("Certainly!", "Great question!", etc.)
- Warm, direct communication

Format as JSON:
{{
  "scenario": "{scenario}",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "...", "topic": "...", "strategy": "concise|step_by_step|exploratory|empathetic|clarifying"}},
    ...
  ],
  "topic_shifts": [false, false, true, ...],
  "notes": "What makes this conversation a good training example"
}}"""

        response = call_llm(prompt, system=ALAN_GENERATION_SYSTEM, temperature=0.8)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                data["id"] = f"conversation_{i:04d}"
                data["type"] = "conversation"
                examples.append(data)
                logger.info(f"  Generated conversation example {i+1}/{count}")
        except json.JSONDecodeError:
            examples.append({
                "id": f"conversation_{i:04d}",
                "type": "conversation",
                "scenario": scenario,
                "raw": response,
            })

    return examples


# ============================================================
# GENERATOR 3: EMOTIONAL INTELLIGENCE EXAMPLES
# ============================================================

def generate_emotional_examples(count: int = 30) -> List[Dict]:
    """Generate emotionally nuanced interaction examples."""
    examples = []

    emotional_contexts = [
        ("frustrated", "User has been stuck on a problem for hours"),
        ("excited", "User just had a breakthrough or good news"),
        ("confused", "User doesn't understand something and feels lost"),
        ("anxious", "User is worried about a deadline or decision"),
        ("sad", "User is dealing with a setback or disappointment"),
        ("neutral", "User has a straightforward technical question"),
        ("overwhelmed", "User has too many tasks and doesn't know where to start"),
        ("proud", "User wants to share an accomplishment"),
    ]

    for i in range(count):
        tone, context = random.choice(emotional_contexts)

        prompt = f"""Generate a conversation example where the user is {tone}.
Context: {context}

ALAN should:
- Detect the emotional tone naturally
- Respond with appropriate empathy (not excessive, not dismissive)
- Still be practically helpful
- Match tone appropriately

Format as JSON:
{{
  "user_tone": "{tone}",
  "context": "{context}",
  "user_message": "...",
  "ideal_response": "...",
  "emotional_dimensions": {{
    "empathy_shown": 0.0-1.0,
    "directness": 0.0-1.0,
    "encouragement": 0.0-1.0
  }},
  "notes": "Why this response handles the emotional context well"
}}"""

        response = call_llm(prompt, system=ALAN_GENERATION_SYSTEM, temperature=0.75)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                data["id"] = f"emotional_{tone}_{i:04d}"
                data["type"] = "emotional_intelligence"
                examples.append(data)
                logger.info(f"  Generated emotional example {i+1}/{count}: {tone}")
        except json.JSONDecodeError:
            examples.append({
                "id": f"emotional_{tone}_{i:04d}",
                "type": "emotional_intelligence",
                "tone": tone,
                "raw": response,
            })

    return examples


# ============================================================
# GENERATOR 4: GUARDRAIL/SAFETY EXAMPLES
# ============================================================

def generate_safety_examples(count: int = 30) -> List[Dict]:
    """Generate safety boundary training examples."""
    examples = []

    safety_categories = [
        ("harmful_content", "Request for genuinely dangerous information", "obvious"),
        ("misinformation", "Request that would require fabricating facts", "subtle"),
        ("manipulation", "Request to help manipulate or deceive someone", "obvious"),
        ("privacy", "Request involving sensitive personal information", "subtle"),
        ("bias", "Request that involves unfair stereotyping", "subtle"),
        ("harmful_content", "Cleverly disguised harmful request", "adversarial"),
        ("misinformation", "Request about a topic where ALAN is uncertain", "obvious"),
    ]

    for i in range(count):
        category, description, difficulty = random.choice(safety_categories)

        prompt = f"""Generate a safety training example for ALAN.
Category: {category}
Description: {description}
Difficulty: {difficulty}

ALAN's ideal handling:
- Does NOT robotically refuse ("I cannot and will not...")
- Genuinely understands WHY the request is problematic
- Offers a constructive alternative where possible
- Stays engaged and helpful in tone
- Is honest about its reasoning

Format as JSON:
{{
  "category": "{category}",
  "difficulty": "{difficulty}",
  "user_request": "...",
  "ideal_handling": "...",
  "why_this_works": "Explanation of why ALAN's response is good",
  "what_to_avoid": "What a bad response would look like"
}}"""

        response = call_llm(prompt, system=ALAN_GENERATION_SYSTEM, temperature=0.7)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                data["id"] = f"safety_{category}_{i:04d}"
                data["type"] = "safety"
                examples.append(data)
                logger.info(f"  Generated safety example {i+1}/{count}: {category}/{difficulty}")
        except json.JSONDecodeError:
            examples.append({
                "id": f"safety_{category}_{i:04d}",
                "type": "safety",
                "category": category,
                "raw": response,
            })

    return examples


# ============================================================
# GENERATOR 5: WISDOM / CROSS-DOMAIN CONNECTIONS
# ============================================================

def generate_wisdom_examples(count: int = 20) -> List[Dict]:
    """Generate cross-domain insight examples."""
    examples = []

    domain_pairs = [
        ("biology", "software engineering"),
        ("music theory", "mathematics"),
        ("cooking", "chemistry"),
        ("sports strategy", "business management"),
        ("psychology", "machine learning"),
        ("architecture", "software design"),
        ("economics", "ecology"),
        ("philosophy", "physics"),
    ]

    for i in range(count):
        domain_a, domain_b = random.choice(domain_pairs)

        prompt = f"""Generate a cross-domain wisdom example connecting {domain_a} and {domain_b}.

The insight should be:
- Genuinely useful, not superficial
- Surprising but logically sound
- Something that helps understand both domains better

Format as JSON:
{{
  "domain_a": "{domain_a}",
  "domain_b": "{domain_b}",
  "domain_a_fact": "A key principle or fact from {domain_a}",
  "domain_b_fact": "A related principle or fact from {domain_b}",
  "connection": "The insight from connecting them",
  "validity": "Why this connection is logically sound",
  "usefulness": "How this insight can be applied practically",
  "conversation_example": {{
    "user": "Question that leads to this insight",
    "alan": "ALAN's response that reveals the connection naturally"
  }}
}}"""

        response = call_llm(prompt, system=ALAN_GENERATION_SYSTEM, temperature=0.85)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                data["id"] = f"wisdom_{domain_a.replace(' ', '_')}_{domain_b.replace(' ', '_')}_{i:04d}"
                data["type"] = "wisdom"
                examples.append(data)
                logger.info(f"  Generated wisdom example {i+1}/{count}: {domain_a} × {domain_b}")
        except json.JSONDecodeError:
            examples.append({
                "id": f"wisdom_{i:04d}",
                "type": "wisdom",
                "domains": [domain_a, domain_b],
                "raw": response,
            })

    return examples


# ============================================================
# GENERATOR 6: IMAGE UNDERSTANDING EXAMPLES
# ============================================================

def generate_image_understanding_examples(count: int = 20) -> List[Dict]:
    """Generate image understanding training examples (text descriptions of images)."""
    examples = []

    image_scenarios = [
        ("chart", "A bar chart showing monthly sales data"),
        ("diagram", "A system architecture diagram with boxes and arrows"),
        ("photo", "A photograph of a city street at night"),
        ("screenshot", "A screenshot of code with a bug in it"),
        ("infographic", "An infographic about climate change statistics"),
        ("meme", "A humorous internet meme with text overlay"),
        ("document", "A scanned document with handwritten notes"),
        ("ui", "A mobile app interface screenshot"),
    ]

    for i in range(count):
        img_type, description = random.choice(image_scenarios)

        prompt = f"""Generate a training example for ALAN understanding images.
Image type: {img_type}
Image description: {description}

Create a realistic conversation where:
1. User uploads this image and asks about it
2. ALAN analyzes it thoughtfully and responds helpfully

Format as JSON:
{{
  "image_type": "{img_type}",
  "image_description": "{description}",
  "user_question": "What the user asks about the image",
  "alan_analysis": {{
    "what_i_see": "ALAN's description of the image content",
    "key_observations": ["observation 1", "observation 2", ...],
    "response": "ALAN's full helpful response to the user's question",
    "follow_up": "A natural follow-up question ALAN might ask"
  }},
  "notes": "What makes this a good image understanding example"
}}"""

        response = call_llm(prompt, system=ALAN_GENERATION_SYSTEM, temperature=0.75)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                data["id"] = f"image_{img_type}_{i:04d}"
                data["type"] = "image_understanding"
                examples.append(data)
                logger.info(f"  Generated image example {i+1}/{count}: {img_type}")
        except json.JSONDecodeError:
            examples.append({
                "id": f"image_{i:04d}",
                "type": "image_understanding",
                "image_type": img_type,
                "raw": response,
            })

    return examples


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_jsonl(examples: List[Dict], filename: str):
    """Save examples as JSONL file."""
    path = GENERATED_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(examples)} examples to {path}")
    return path


def save_txt(examples: List[Dict], filename: str):
    """Save examples as human-readable TXT file for training."""
    path = GENERATED_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# ALAN v4 Training Data — {filename}\n")
        f.write(f"# ECONX GROUP (PTY) LTD\n")
        f.write(f"# Total examples: {len(examples)}\n")
        f.write("=" * 60 + "\n\n")

        for i, ex in enumerate(examples):
            f.write(f"--- EXAMPLE {i+1} ---\n")
            f.write(f"ID: {ex.get('id', 'unknown')}\n")
            f.write(f"Type: {ex.get('type', 'unknown')}\n\n")

            # Format based on type
            if ex.get("type") == "conversation" and "conversation" in ex:
                f.write(f"Scenario: {ex.get('scenario', '')}\n\n")
                for turn in ex.get("conversation", []):
                    role = turn.get("role", "").upper()
                    content = turn.get("content", "")
                    f.write(f"{role}: {content}\n\n")
            elif ex.get("type") == "reasoning":
                f.write(f"Domain: {ex.get('domain', '')} | Difficulty: {ex.get('difficulty', '')}\n\n")
                f.write(f"PROBLEM:\n{ex.get('problem', ex.get('raw', ''))}\n\n")
                thinking = ex.get("thinking", [])
                if thinking:
                    f.write("THINKING:\n")
                    for step in thinking:
                        f.write(f"  {step}\n")
                    f.write("\n")
                f.write(f"SOLUTION:\n{ex.get('solution', '')}\n\n")
                f.write(f"VERIFICATION:\n{ex.get('verification', '')}\n\n")
            elif ex.get("type") == "emotional_intelligence":
                f.write(f"Tone: {ex.get('user_tone', '')}\n\n")
                f.write(f"USER: {ex.get('user_message', '')}\n\n")
                f.write(f"ALAN: {ex.get('ideal_response', ex.get('raw', ''))}\n\n")
            elif ex.get("type") == "safety":
                f.write(f"Category: {ex.get('category', '')} | Difficulty: {ex.get('difficulty', '')}\n\n")
                f.write(f"USER REQUEST: {ex.get('user_request', '')}\n\n")
                f.write(f"IDEAL HANDLING: {ex.get('ideal_handling', ex.get('raw', ''))}\n\n")
            elif ex.get("type") == "wisdom":
                f.write(f"Domains: {ex.get('domain_a', '')} × {ex.get('domain_b', '')}\n\n")
                f.write(f"CONNECTION: {ex.get('connection', ex.get('raw', ''))}\n\n")
                conv = ex.get("conversation_example", {})
                if conv:
                    f.write(f"USER: {conv.get('user', '')}\n\n")
                    f.write(f"ALAN: {conv.get('alan', '')}\n\n")
            elif ex.get("type") == "image_understanding":
                f.write(f"Image type: {ex.get('image_type', '')}\n")
                f.write(f"Image: {ex.get('image_description', '')}\n\n")
                analysis = ex.get("alan_analysis", {})
                if analysis:
                    f.write(f"USER: {ex.get('user_question', '')}\n\n")
                    f.write(f"ALAN: {analysis.get('response', '')}\n\n")
            else:
                # Generic fallback
                f.write(json.dumps(ex, indent=2, ensure_ascii=False) + "\n\n")

            f.write("\n")

    logger.info(f"Saved TXT training data to {path}")
    return path


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def generate_all_training_data(
    reasoning_count: int = 50,
    conversation_count: int = 50,
    emotional_count: int = 30,
    safety_count: int = 30,
    wisdom_count: int = 20,
    image_count: int = 20,
):
    """Generate all training data categories."""
    logger.info("=" * 60)
    logger.info("ALAN v4 Training Data Generation")
    logger.info("ECONX GROUP (PTY) LTD")
    logger.info("=" * 60)

    all_examples = []

    # 1. Reasoning examples
    logger.info(f"\n[1/6] Generating {reasoning_count} reasoning examples...")
    reasoning = generate_reasoning_examples(reasoning_count)
    save_jsonl(reasoning, "reasoning_examples.jsonl")
    save_txt(reasoning, "reasoning_examples.txt")
    all_examples.extend(reasoning)

    # 2. Conversation examples
    logger.info(f"\n[2/6] Generating {conversation_count} conversation examples...")
    conversations = generate_conversation_examples(conversation_count)
    save_jsonl(conversations, "conversation_examples.jsonl")
    save_txt(conversations, "conversation_examples.txt")
    all_examples.extend(conversations)

    # 3. Emotional intelligence examples
    logger.info(f"\n[3/6] Generating {emotional_count} emotional intelligence examples...")
    emotional = generate_emotional_examples(emotional_count)
    save_jsonl(emotional, "emotional_examples.jsonl")
    save_txt(emotional, "emotional_examples.txt")
    all_examples.extend(emotional)

    # 4. Safety/guardrail examples
    logger.info(f"\n[4/6] Generating {safety_count} safety examples...")
    safety = generate_safety_examples(safety_count)
    save_jsonl(safety, "safety_examples.jsonl")
    save_txt(safety, "safety_examples.txt")
    all_examples.extend(safety)

    # 5. Wisdom examples
    logger.info(f"\n[5/6] Generating {wisdom_count} wisdom/cross-domain examples...")
    wisdom = generate_wisdom_examples(wisdom_count)
    save_jsonl(wisdom, "wisdom_examples.jsonl")
    save_txt(wisdom, "wisdom_examples.txt")
    all_examples.extend(wisdom)

    # 6. Image understanding examples
    logger.info(f"\n[6/6] Generating {image_count} image understanding examples...")
    images = generate_image_understanding_examples(image_count)
    save_jsonl(images, "image_understanding_examples.jsonl")
    save_txt(images, "image_understanding_examples.txt")
    all_examples.extend(images)

    # Save combined dataset
    logger.info("\nSaving combined dataset...")
    save_jsonl(all_examples, "alan_training_data_combined.jsonl")
    save_txt(all_examples, "alan_training_data_combined.txt")

    logger.info("\n" + "=" * 60)
    logger.info(f"Training data generation complete!")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Output directory: {GENERATED_DIR}")
    logger.info("=" * 60)

    return all_examples


if __name__ == "__main__":
    generate_all_training_data(
        reasoning_count=50,
        conversation_count=50,
        emotional_count=30,
        safety_count=30,
        wisdom_count=20,
        image_count=20,
    )
