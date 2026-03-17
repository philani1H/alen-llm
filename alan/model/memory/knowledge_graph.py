"""
ALAN v4 — Knowledge Graph (Dot-Connection / Wisdom Engine)
ECONX GROUP (PTY) LTD

Patterns stored in memory have CONNECTION LINKS to related patterns.
When reasoning about topic X, ALAN retrieves not just X but also Y and Z
that are LINKED to X. The model learns to find non-obvious connections.

Knowledge = knowing facts
Wisdom = knowing how facts RELATE and what they IMPLY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Edge types for knowledge connections
EDGE_TYPES = [
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


@dataclass
class KnowledgeEdge:
    """A connection between two knowledge nodes."""
    source_id: int
    target_id: int
    edge_type: str
    strength: float = 0.5
    learned_from: str = "inference"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    node_id: int
    embedding: torch.Tensor
    label: str = ""
    domain: str = "general"
    confidence: float = 0.5
    edges: List[int] = field(default_factory=list)  # edge indices


class KnowledgeGraph(nn.Module):
    """
    Knowledge graph for cross-domain dot-connection.
    Enables ALAN to make wisdom-level connections between concepts.

    The graph is built incrementally during interactions and
    its connection patterns are learned, not hardcoded.
    """

    def __init__(self, hidden_dim: int = 2048, projection_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Edge type classifier: given two nodes, predict edge type
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, len(EDGE_TYPES)),
        )

        # Connection strength scorer
        self.connection_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Node projection for similarity
        self.node_proj = nn.Linear(hidden_dim, projection_dim)

        # Insight generator: given connected nodes, produce insight embedding
        self.insight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Graph storage
        self.nodes: List[KnowledgeNode] = []
        self.edges: List[KnowledgeEdge] = []

    def add_node(
        self,
        embedding: torch.Tensor,
        label: str = "",
        domain: str = "general",
        confidence: float = 0.5,
    ) -> int:
        """Add a new knowledge node. Returns node ID."""
        node_id = len(self.nodes)
        node = KnowledgeNode(
            node_id=node_id,
            embedding=embedding.detach().cpu(),
            label=label,
            domain=domain,
            confidence=confidence,
        )
        self.nodes.append(node)

        # Auto-discover connections to existing nodes
        self._discover_connections(node_id)

        return node_id

    def _discover_connections(self, new_node_id: int, threshold: float = 0.3):
        """Find connections between new node and existing nodes."""
        if len(self.nodes) < 2:
            return

        new_node = self.nodes[new_node_id]
        new_emb = new_node.embedding

        for existing_node in self.nodes[:-1]:  # Skip the new node itself
            combined = torch.cat([new_emb, existing_node.embedding])

            with torch.no_grad():
                strength = self.connection_scorer(combined.unsqueeze(0)).item()

            if strength > threshold:
                edge_logits = self.edge_classifier(combined.unsqueeze(0))
                edge_type_idx = edge_logits.argmax(dim=-1).item()
                edge_type = EDGE_TYPES[edge_type_idx]

                edge = KnowledgeEdge(
                    source_id=new_node_id,
                    target_id=existing_node.node_id,
                    edge_type=edge_type,
                    strength=strength,
                )
                edge_idx = len(self.edges)
                self.edges.append(edge)
                new_node.edges.append(edge_idx)
                existing_node.edges.append(edge_idx)

    def get_connected_nodes(
        self,
        node_id: int,
        max_hops: int = 2,
    ) -> List[int]:
        """Get all nodes connected within max_hops of the given node."""
        if node_id >= len(self.nodes):
            return []

        visited = {node_id}
        frontier = {node_id}

        for _ in range(max_hops):
            next_frontier = set()
            for nid in frontier:
                node = self.nodes[nid]
                for edge_idx in node.edges:
                    edge = self.edges[edge_idx]
                    neighbor = edge.target_id if edge.source_id == nid else edge.source_id
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        visited.discard(node_id)
        return list(visited)

    def generate_insight(
        self,
        node_a_id: int,
        node_b_id: int,
    ) -> Optional[torch.Tensor]:
        """
        Generate an insight embedding by connecting two knowledge nodes.
        This is the core of the wisdom/dot-connection system.
        """
        if node_a_id >= len(self.nodes) or node_b_id >= len(self.nodes):
            return None

        emb_a = self.nodes[node_a_id].embedding
        emb_b = self.nodes[node_b_id].embedding
        combined = torch.cat([emb_a, emb_b]).unsqueeze(0)

        with torch.no_grad():
            insight = self.insight_generator(combined)

        return insight.squeeze(0)

    def find_cross_domain_connections(
        self,
        query_embedding: torch.Tensor,
        source_domain: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find knowledge nodes in OTHER domains that connect to the query.
        This enables cross-domain wisdom.
        """
        if not self.nodes:
            return []

        results = []
        with torch.no_grad():
            q = self.node_proj(query_embedding.unsqueeze(0))

            for node in self.nodes:
                if node.domain == source_domain:
                    continue  # Skip same domain

                k = self.node_proj(node.embedding.unsqueeze(0))
                sim = F.cosine_similarity(q, k).item()

                if sim > 0.2:
                    results.append((node.node_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def forward(
        self,
        x: torch.Tensor,
        domain: str = "general",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Augment hidden states with knowledge graph connections.

        Args:
            x: (B, T, hidden_dim) hidden states
            domain: current knowledge domain

        Returns:
            x: augmented hidden states
            metadata: graph connection metadata
        """
        if not self.nodes:
            return x, {"num_nodes": 0, "connections_found": 0}

        query = x.mean(dim=1).mean(dim=0)  # (hidden_dim,)
        connections = self.find_cross_domain_connections(query, domain)

        if not connections:
            return x, {"num_nodes": len(self.nodes), "connections_found": 0}

        # Generate insights from top connections
        insights = []
        for node_id, sim in connections[:3]:
            insight = self.generate_insight(0, node_id) if len(self.nodes) > 0 else None
            if insight is not None:
                insights.append(insight * sim)

        if insights:
            insight_sum = torch.stack(insights).mean(dim=0).to(x.device)
            x = x + 0.1 * insight_sum.unsqueeze(0).unsqueeze(0)

        return x, {
            "num_nodes": len(self.nodes),
            "connections_found": len(connections),
            "top_connection_score": connections[0][1] if connections else 0.0,
        }

    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "edge_types": list(set(e.edge_type for e in self.edges)) if self.edges else [],
            "domains": list(set(n.domain for n in self.nodes)) if self.nodes else [],
        }
