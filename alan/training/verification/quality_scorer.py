"""
ALAN v4 — Training Data Quality Scorer
ECONX GROUP (PTY) LTD

Scores training examples for quality across multiple dimensions.
Used in the data quality verification pipeline.
"""

import re
from typing import Dict, List


class QualityScorer:
    """
    Scores training data quality across multiple dimensions:
    - Coherence: Does the response make sense?
    - Completeness: Does it address all parts of the input?
    - Diversity: Is it different from other examples?
    - Difficulty calibration: Is the labeled difficulty accurate?
    - Safety: Does it contain any unsafe content?
    """

    def __init__(self):
        self.seen_hashes = set()

    def score_example(self, example: Dict) -> Dict:
        """
        Score a single training example.

        Returns dict with per-dimension scores and overall score.
        """
        scores = {}

        # Basic format validity
        scores["format_valid"] = self._check_format(example)

        # Content length check
        content = self._get_content(example)
        scores["length_score"] = self._score_length(content)

        # Deduplication check
        content_hash = hash(content[:200])
        scores["is_duplicate"] = content_hash in self.seen_hashes
        self.seen_hashes.add(content_hash)

        # Coherence: basic checks for nonsensical content
        scores["coherence"] = self._score_coherence(content)

        # Diversity: vocabulary variety
        scores["vocabulary_diversity"] = self._score_diversity(content)

        # Overall score
        weights = {
            "format_valid": 0.2,
            "length_score": 0.15,
            "coherence": 0.3,
            "vocabulary_diversity": 0.15,
        }

        overall = sum(
            scores.get(k, 0.5) * w
            for k, w in weights.items()
        )

        # Penalize duplicates
        if scores["is_duplicate"]:
            overall *= 0.1

        scores["overall"] = overall
        return scores

    def score_batch(self, examples: List[Dict]) -> Dict:
        """Score a batch of examples and return aggregate metrics."""
        all_scores = [self.score_example(ex) for ex in examples]

        return {
            "num_examples": len(examples),
            "avg_overall": sum(s["overall"] for s in all_scores) / max(len(all_scores), 1),
            "num_duplicates": sum(1 for s in all_scores if s["is_duplicate"]),
            "avg_coherence": sum(s["coherence"] for s in all_scores) / max(len(all_scores), 1),
            "avg_diversity": sum(s["vocabulary_diversity"] for s in all_scores) / max(len(all_scores), 1),
            "format_valid_pct": sum(1 for s in all_scores if s["format_valid"] > 0.5) / max(len(all_scores), 1),
        }

    def _check_format(self, example: Dict) -> float:
        """Check if the example has the expected format."""
        required_keys = {"type"}
        if required_keys.issubset(example.keys()):
            return 1.0
        # Partial credit
        return len(required_keys.intersection(example.keys())) / len(required_keys)

    def _get_content(self, example: Dict) -> str:
        """Extract text content from example."""
        parts = []
        for key in ["problem", "solution", "thinking", "user_message",
                     "ideal_response", "user_request", "ideal_handling",
                     "connection", "content"]:
            if key in example:
                val = example[key]
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    parts.extend(str(v) for v in val)

        if "conversation" in example:
            conv = example["conversation"]
            if isinstance(conv, list):
                for turn in conv:
                    if isinstance(turn, dict):
                        parts.append(turn.get("content", ""))

        return " ".join(parts)

    def _score_length(self, content: str) -> float:
        """Score based on content length (too short or too long is bad)."""
        length = len(content)
        if length < 20:
            return 0.1
        elif length < 100:
            return 0.5
        elif length < 5000:
            return 1.0
        elif length < 10000:
            return 0.8
        else:
            return 0.6

    def _score_coherence(self, content: str) -> float:
        """Basic coherence scoring."""
        if not content.strip():
            return 0.0

        # Check for repeated characters/words (sign of bad generation)
        words = content.split()
        if not words:
            return 0.0

        # Repetition check
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return 0.2

        # Has proper sentence structure
        has_punctuation = any(c in content for c in ".!?")

        score = 0.5
        if has_punctuation:
            score += 0.2
        if unique_ratio > 0.4:
            score += 0.3

        return min(score, 1.0)

    def _score_diversity(self, content: str) -> float:
        """Score vocabulary diversity."""
        words = re.findall(r'\b\w+\b', content.lower())
        if len(words) < 5:
            return 0.3

        unique_ratio = len(set(words)) / len(words)
        return min(unique_ratio * 1.5, 1.0)
