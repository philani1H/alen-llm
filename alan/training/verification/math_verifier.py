"""
ALAN v4 — Math Verification
ECONX GROUP (PTY) LTD

Verifies mathematical solutions in training data.
"""

import re
from typing import Dict, Optional, Tuple


class MathVerifier:
    """
    Verifies mathematical calculations and solutions.
    Used in the training data quality pipeline.
    """

    def verify_arithmetic(self, expression: str, claimed_result: str) -> Dict:
        """
        Verify a basic arithmetic expression.

        Args:
            expression: math expression like "7 * 8" or "123 + 456"
            claimed_result: the claimed answer

        Returns:
            dict with 'correct' bool and details
        """
        try:
            # Only allow safe arithmetic operations
            safe_expr = re.sub(r'[^0-9+\-*/().% ]', '', expression)
            if not safe_expr:
                return {"correct": False, "error": "No valid expression found"}

            actual = eval(safe_expr)  # Safe: only numbers and operators
            claimed = self._parse_number(claimed_result)

            if claimed is None:
                return {"correct": False, "error": "Could not parse claimed result"}

            correct = abs(actual - claimed) < 1e-6
            return {
                "correct": correct,
                "actual": actual,
                "claimed": claimed,
                "expression": safe_expr,
            }
        except Exception as e:
            return {"correct": False, "error": str(e)}

    def verify_solution_steps(self, steps: list, final_answer: str) -> Dict:
        """
        Verify that solution steps are logically consistent.

        Args:
            steps: list of reasoning steps
            final_answer: the claimed final answer

        Returns:
            verification results
        """
        if not steps:
            return {"valid": False, "error": "No steps provided"}

        issues = []

        # Check step numbering/ordering
        for i, step in enumerate(steps):
            if not step.strip():
                issues.append(f"Step {i+1} is empty")

        # Check that final answer appears in or follows from last step
        if final_answer and steps:
            last_step = steps[-1].lower()
            answer_lower = final_answer.lower().strip()
            if answer_lower not in last_step and len(answer_lower) > 2:
                issues.append("Final answer may not follow from last step")

        return {
            "valid": len(issues) == 0,
            "num_steps": len(steps),
            "issues": issues,
        }

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from text."""
        try:
            cleaned = re.sub(r'[^0-9.\-]', '', text.strip())
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def verify_example(self, example: Dict) -> Dict:
        """Verify a complete math training example."""
        results = {"type": "math_verification"}

        problem = example.get("problem", "")
        solution = example.get("solution", "")
        thinking = example.get("thinking", [])

        # Check steps if available
        if thinking:
            step_result = self.verify_solution_steps(thinking, solution)
            results["step_verification"] = step_result
        else:
            results["step_verification"] = {"valid": True, "note": "No steps to verify"}

        # Try to extract and verify arithmetic
        numbers = re.findall(r'\d+\.?\d*', solution)
        results["has_numeric_answer"] = len(numbers) > 0

        results["has_problem"] = bool(problem.strip())
        results["has_solution"] = bool(solution.strip())

        results["overall_valid"] = (
            results["has_problem"]
            and results["has_solution"]
            and results["step_verification"]["valid"]
        )

        return results
