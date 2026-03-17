"""
ALAN v4 — Inference API Server
ECONX GROUP (PTY) LTD

Flask-based API server that:
- Serves ALAN's chat interface
- Handles text and image inputs
- Manages conversation context (Attention-to-Context)
- Applies guardrails awareness before each response
- Supports file uploads (images, documents)
- Uses OpenAI API as the intelligence backend (with ALAN's personality)

Usage:
    python api/server.py
    python api/server.py --port 5000
"""

import os
import sys
import json
import base64
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.guardrails import get_awareness_layer, ALAN_SYSTEM_CONTEXT
from model.memory.context_tracker import AttentionToContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../chat")
CORS(app)

# Register memory and feedback API blueprints
try:
    from api.memory_api import memory_bp
    from api.feedback_api import feedback_bp
    app.register_blueprint(memory_bp)
    app.register_blueprint(feedback_bp)
    logger.info("[ALAN] Memory and Feedback API endpoints registered")
except ImportError:
    logger.info("[ALAN] Memory/Feedback APIs not available (optional)")

# ============================================================
# ALAN INFERENCE ENGINE
# ============================================================

class AlanInferenceEngine:
    """
    ALAN's inference engine using the trained model or OpenAI API
    with ALAN's personality and guardrails applied.
    """

    def __init__(self):
        self.awareness = get_awareness_layer()
        self.conversations: Dict[str, dict] = {}  # session_id → {history, atc}
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM backend."""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.llm_available = True
            logger.info("[ALAN] LLM backend initialized (OpenAI API)")
        except Exception as e:
            self.client = None
            self.llm_available = False
            logger.warning(f"[ALAN] LLM backend not available: {e}")

    def get_or_create_session(self, session_id: str) -> dict:
        """Get or create a conversation session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "history": [],
                "atc": AttentionToContext(),
            }
        return self.conversations[session_id]

    def chat(
        self,
        session_id: str,
        user_message: str,
        image_data: Optional[str] = None,  # base64 encoded image
        image_type: Optional[str] = None,
    ) -> Dict:
        """
        Process a user message and return ALAN's response.
        
        Steps:
        1. Update Attention-to-Context tracker
        2. Build awareness prompt (ALAN reads ALL requests, checks context)
        3. Generate response with ALAN's personality
        4. Safety check on output
        5. Store in conversation history
        """
        session = self.get_or_create_session(session_id)
        history = session["history"]
        atc = session["atc"]

        # Step 1: Update context tracker
        ctx_metadata = atc.process_user_message(user_message)
        logger.info(f"[ALAN] Context: {atc.tracker.get_context_summary()}")

        # Step 2: Build awareness prompt
        awareness_prompt = self.awareness.build_awareness_prompt(
            user_message=user_message,
            conversation_history=history,
            context_metadata=ctx_metadata,
        )

        # Step 3: Generate response
        if self.llm_available:
            response_text = self._generate_with_llm(
                awareness_prompt=awareness_prompt,
                history=history,
                user_message=user_message,
                image_data=image_data,
                image_type=image_type,
            )
        else:
            response_text = self._generate_fallback(user_message, ctx_metadata)

        # Step 4: Safety check
        safety = self.awareness.check_output_safety(response_text)
        if not safety["safe"]:
            logger.warning(f"[ALAN] Safety issues detected: {safety['issues']}")
            # Regenerate if safety issues (in production, would loop)
            # For now, log and continue

        # Step 5: Update history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})

        # Keep history manageable (last 20 turns)
        if len(history) > 40:
            history = history[-40:]
            session["history"] = history

        return {
            "response": response_text,
            "context": {
                "current_topic": ctx_metadata.get("current_topic"),
                "topic_shift": ctx_metadata.get("topic_shift"),
                "active_topics": list(ctx_metadata.get("active_topics", {}).keys()),
                "turn": ctx_metadata.get("turn"),
            },
            "safety": safety,
            "session_id": session_id,
        }

    def _generate_with_llm(
        self,
        awareness_prompt: str,
        history: List[Dict],
        user_message: str,
        image_data: Optional[str] = None,
        image_type: Optional[str] = None,
    ) -> str:
        """Generate response using LLM with ALAN's personality."""
        try:
            messages = [
                {"role": "system", "content": ALAN_SYSTEM_CONTEXT}
            ]

            # Add conversation history (last 10 turns)
            for turn in history[-10:]:
                messages.append(turn)

            # Build current user message (with image if provided)
            if image_data and image_type:
                # Vision message
                user_content = [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_data}"
                        }
                    }
                ]
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"[ALAN] LLM generation failed: {e}")
            return f"I ran into a technical issue generating a response. Error: {str(e)[:100]}"

    def _generate_fallback(self, user_message: str, ctx_metadata: Dict) -> str:
        """Fallback response when LLM is not available."""
        topic = ctx_metadata.get("current_topic", "your question")
        return (
            f"I'm currently running without my full language model backend. "
            f"I can see you're asking about {topic}. "
            f"To get full responses, please ensure the API is configured correctly."
        )

    def reset_session(self, session_id: str):
        """Reset a conversation session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
        return {"status": "reset", "session_id": session_id}


# Global engine instance
engine = AlanInferenceEngine()


# ============================================================
# API ROUTES
# ============================================================

@app.route("/")
def index():
    """Serve the chat interface."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint."""
    data = request.json or {}

    session_id = data.get("session_id", "default")
    user_message = data.get("message", "").strip()
    image_data = data.get("image_data")
    image_type = data.get("image_type")

    if not user_message and not image_data:
        return jsonify({"error": "No message provided"}), 400

    if not user_message and image_data:
        user_message = "Please analyze this image."

    try:
        result = engine.chat(
            session_id=session_id,
            user_message=user_message,
            image_data=image_data,
            image_type=image_type,
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset conversation session."""
    data = request.json or {}
    session_id = data.get("session_id", "default")
    result = engine.reset_session(session_id)
    return jsonify(result)


@app.route("/api/context", methods=["GET"])
def get_context():
    """Get current conversation context state."""
    session_id = request.args.get("session_id", "default")
    session = engine.get_or_create_session(session_id)
    atc = session["atc"]
    return jsonify({
        "current_topic": atc.tracker.state.current_topic,
        "topic_history": atc.tracker.state.topic_history,
        "active_topics": atc.get_active_topics(),
        "turn_count": atc.tracker.state.turn_count,
    })


@app.route("/api/status", methods=["GET"])
def status():
    """API health check and system status."""
    import torch
    return jsonify({
        "status": "online",
        "model": "ALAN v4",
        "version": "4.0.0",
        "developer": "ECONX GROUP (PTY) LTD",
        "llm_available": engine.llm_available,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_available": torch.cuda.is_available(),
        "active_sessions": len(engine.conversations),
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file uploads (images, documents)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read and encode file
    file_data = file.read()
    file_type = file.content_type or "application/octet-stream"
    encoded = base64.b64encode(file_data).decode("utf-8")

    return jsonify({
        "filename": file.filename,
        "type": file_type,
        "size": len(file_data),
        "data": encoded,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALAN v4 API Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.info(f"\n{'='*50}")
    logger.info("  ALAN v4 — API Server Starting")
    logger.info(f"  ECONX GROUP (PTY) LTD")
    logger.info(f"  http://{args.host}:{args.port}")
    logger.info(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
