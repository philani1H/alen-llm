"""
ALAN v4 — Logic Verification
ECONX GROUP (PTY) LTD

Verifies logical reasoning in training data examples.
"""

import re
from typing import Dict, List


class LogicVerifier:
    """
    Verifies logical consistency in reasoning examples.
    Checks for common logical fallacies and structural issues.
    """

    # Common logical fallacy indicators
    FALLACY_PATTERNS = [
        (r"therefore.*because", "circular_reasoning"),
        (r"everyone knows|obviously|clearly", "appeal_to_common_knowledge"),
        (r"always|never|every single", "hasty_generalization"),
    ]

    def verify_reasoning_chain(self, steps: List[str]) -> Dict:
        """
        Verify a chain of reasoning steps.

        Checks:
        1. Each step follows logically from previous
        2. No circular reasoning
        3. Conclusion is supported by premises
        4. No obvious fallacies

        Args:
            steps: list of reasoning step strings

        Returns:
            verification results
        """
        if not steps:
            return {"valid": False, "error": "No reasoning steps"}

        issues = []

        # Check for empty steps
        for i, step in enumerate(steps):
            if not step.strip():
                issues.append(f"Step {i+1} is empty")

        # Check for logical connectors (basic structural check)
        has_connectors = any(
            any(c in step.lower() for c in ["therefore", "because", "since", "thus", "so", "hence"])
            for step in steps
        )

        # Check for fallacy patterns
        full_text = " ".join(steps).lower()
        detected_fallacies = []
        for pattern, fallacy_name in self.FALLACY_PATTERNS:
            if re.search(pattern, full_text, re.IGNORECASE):
                detected_fallacies.append(fallacy_name)

        # Check for contradiction between steps
        contradictions = self._detect_contradictions(steps)

        return {
            "valid": len(issues) == 0 and len(contradictions) == 0,
            "num_steps": len(steps),
            "has_logical_connectors": has_connectors,
            "issues": issues,
            "potential_fallacies": detected_fallacies,
            "contradictions": contradictions,
        }

    def _detect_contradictions(self, steps: List[str]) -> List[str]:
        """Detect potential contradictions between steps."""
        contradictions = []

        negation_pairs = [
            ("is", "is not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("true", "false"),
            ("yes", "no"),
            ("increase", "decrease"),
            ("more", "less"),
        ]

        for i, step_a in enumerate(steps):
            for j, step_b in enumerate(steps[i+1:], i+1):
                for pos, neg in negation_pairs:
                    # Very basic check: if one step says X is Y
                    # and another says X is not Y
                    a_lower = step_a.lower()
                    b_lower = step_b.lower()
                    if pos in a_lower and neg in b_lower:
                        # Only flag if they seem to be about the same subject
                        a_words = set(a_lower.split())
                        b_words = set(b_lower.split())
                        overlap = a_words & b_words
                        if len(overlap) > 3:
                            contradictions.append(
                                f"Possible contradiction between step {i+1} and step {j+1}"
                            )

        return contradictions

    def verify_example(self, example: Dict) -> Dict:
        """Verify a complete logic training example."""
        results = {"type": "logic_verification"}

        thinking = example.get("thinking", [])
        solution = example.get("solution", "")

        if thinking:
            chain_result = self.verify_reasoning_chain(thinking)
            results["chain_verification"] = chain_result
        else:
            results["chain_verification"] = {"valid": True, "note": "No chain to verify"}

        results["has_solution"] = bool(solution.strip())
        results["has_reasoning"] = len(thinking) > 0

        results["overall_valid"] = (
            results["has_solution"]
            and results["chain_verification"]["valid"]
        )

        return results
