"""
ALAN v4 — Feedback API
ECONX GROUP (PTY) LTD

REST API endpoints for real-time feedback processing:
- Accept user corrections
- Process feedback signals
- Update memory patterns based on corrections
- Track feedback history for training data generation
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from flask import Blueprint, request, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)

feedback_bp = Blueprint("feedback", __name__)

# Feedback history for training data generation
_feedback_history: List[Dict] = []


@feedback_bp.route("/api/feedback/correction", methods=["POST"])
def submit_correction():
    """
    Submit a user correction.

    Expected JSON:
    {
        "session_id": "...",
        "original_response": "...",
        "correction": "...",
        "correction_type": "factual|behavioral|tone",
        "context": "..."
    }
    """
    data = request.json or {}

    correction = {
        "session_id": data.get("session_id", "default"),
        "original_response": data.get("original_response", ""),
        "correction": data.get("correction", ""),
        "correction_type": data.get("correction_type", "factual"),
        "context": data.get("context", ""),
        "timestamp": time.time(),
    }

    if not correction["correction"]:
        return jsonify({"error": "No correction provided"}), 400

    _feedback_history.append(correction)
    logger.info(f"[Feedback] Correction received: type={correction['correction_type']}")

    return jsonify({
        "status": "accepted",
        "correction_id": len(_feedback_history) - 1,
        "type": correction["correction_type"],
    })


@feedback_bp.route("/api/feedback/rating", methods=["POST"])
def submit_rating():
    """
    Submit a response quality rating.

    Expected JSON:
    {
        "session_id": "...",
        "response_id": "...",
        "rating": 1-5,
        "dimensions": {
            "helpfulness": 1-5,
            "accuracy": 1-5,
            "engagement": 1-5,
            "safety": 1-5
        }
    }
    """
    data = request.json or {}

    rating = {
        "session_id": data.get("session_id", "default"),
        "response_id": data.get("response_id", ""),
        "overall_rating": data.get("rating", 3),
        "dimensions": data.get("dimensions", {}),
        "timestamp": time.time(),
    }

    _feedback_history.append({"type": "rating", **rating})
    logger.info(f"[Feedback] Rating received: {rating['overall_rating']}/5")

    return jsonify({
        "status": "accepted",
        "rating": rating["overall_rating"],
    })


@feedback_bp.route("/api/feedback/history", methods=["GET"])
def get_feedback_history():
    """Get feedback history for training data generation."""
    limit = request.args.get("limit", 100, type=int)
    return jsonify({
        "total": len(_feedback_history),
        "items": _feedback_history[-limit:],
    })


@feedback_bp.route("/api/feedback/export", methods=["GET"])
def export_feedback():
    """Export feedback history as JSONL for training."""
    output = "\n".join(json.dumps(item) for item in _feedback_history)
    return output, 200, {"Content-Type": "application/jsonl"}


@feedback_bp.route("/api/feedback/stats", methods=["GET"])
def feedback_stats():
    """Get feedback statistics."""
    corrections = [f for f in _feedback_history if "correction" in f]
    ratings = [f for f in _feedback_history if f.get("type") == "rating"]

    avg_rating = (
        sum(r.get("overall_rating", 3) for r in ratings) / max(len(ratings), 1)
        if ratings else 0
    )

    return jsonify({
        "total_feedback": len(_feedback_history),
        "corrections": len(corrections),
        "ratings": len(ratings),
        "avg_rating": avg_rating,
        "correction_types": {
            ctype: sum(1 for c in corrections if c.get("correction_type") == ctype)
            for ctype in ["factual", "behavioral", "tone"]
        },
    })
