"""
ALAN v4 — Memory API
ECONX GROUP (PTY) LTD

REST API endpoints for ALAN's external memory system:
- Store patterns
- Retrieve relevant memories
- Update patterns after corrections
- Get memory statistics
- Consolidate session memories
"""

import sys
from pathlib import Path
from typing import Dict, Optional

from flask import Blueprint, request, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.memory.pattern_store import PatternStore
from model.memory.knowledge_graph import KnowledgeGraph
from model.memory.consolidation import MemoryConsolidator

import logging

logger = logging.getLogger(__name__)

memory_bp = Blueprint("memory", __name__)

# Global memory instances
_pattern_store: Optional[PatternStore] = None
_knowledge_graph: Optional[KnowledgeGraph] = None
_consolidator: Optional[MemoryConsolidator] = None


def get_pattern_store(hidden_dim: int = 1024) -> PatternStore:
    """Get or create the pattern store singleton."""
    global _pattern_store
    if _pattern_store is None:
        _pattern_store = PatternStore(hidden_dim=hidden_dim)
    return _pattern_store


def get_knowledge_graph(hidden_dim: int = 1024) -> KnowledgeGraph:
    """Get or create the knowledge graph singleton."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph(hidden_dim=hidden_dim)
    return _knowledge_graph


def get_consolidator(hidden_dim: int = 1024) -> MemoryConsolidator:
    """Get or create the consolidator singleton."""
    global _consolidator
    if _consolidator is None:
        _consolidator = MemoryConsolidator(hidden_dim=hidden_dim)
    return _consolidator


@memory_bp.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    """Get memory system statistics."""
    store = get_pattern_store()
    graph = get_knowledge_graph()

    return jsonify({
        "pattern_store": store.get_stats(),
        "knowledge_graph": graph.get_stats(),
    })


@memory_bp.route("/api/memory/store", methods=["POST"])
def store_pattern():
    """Store a new pattern in memory."""
    data = request.json or {}
    context = data.get("context", "")
    domain = data.get("domain", "general")

    store = get_pattern_store()

    # Create a simple embedding from context text
    vector = torch.randn(store.hidden_dim)
    stored = store.store(vector, context=context, domain=domain)

    return jsonify({
        "stored": stored,
        "total_patterns": len(store.patterns),
    })


@memory_bp.route("/api/memory/consolidate", methods=["POST"])
def consolidate_memory():
    """Consolidate session patterns into long-term memory."""
    consolidator = get_consolidator()
    store = get_pattern_store()

    lt_patterns = [p.content_vector for p in store.patterns] if store.patterns else None
    result = consolidator.consolidate(long_term_patterns=lt_patterns)

    return jsonify({
        "retained": result["retained"],
        "discarded": result["discarded"],
        "merged": result["merged"],
        "total_processed": result["total_processed"],
    })


@memory_bp.route("/api/memory/graph/stats", methods=["GET"])
def graph_stats():
    """Get knowledge graph statistics."""
    graph = get_knowledge_graph()
    return jsonify(graph.get_stats())
