"""
ALAN v4 — Code Verification
ECONX GROUP (PTY) LTD

Verifies code solutions in training data by checking syntax validity.
"""

import ast
import re
from typing import Dict, List


class CodeVerifier:
    """
    Verifies code examples in training data.
    Checks syntax validity, structure, and basic quality.
    """

    def verify_python_syntax(self, code: str) -> Dict:
        """
        Verify Python code syntax validity.

        Args:
            code: Python code string

        Returns:
            dict with 'valid' bool and details
        """
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
            }

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text (markdown-style or indented)."""
        blocks = []

        # Markdown code blocks
        pattern = r'```(?:python)?\s*\n(.*?)\n```'
        for match in re.finditer(pattern, text, re.DOTALL):
            blocks.append(match.group(1).strip())

        # If no markdown blocks, look for indented blocks
        if not blocks:
            lines = text.split('\n')
            current_block = []
            for line in lines:
                if line.startswith('    ') or line.startswith('\t'):
                    current_block.append(line)
                elif current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
            if current_block:
                blocks.append('\n'.join(current_block))

        return blocks

    def check_code_quality(self, code: str) -> Dict:
        """
        Basic code quality checks.

        Returns quality metrics.
        """
        lines = code.strip().split('\n')
        non_empty = [l for l in lines if l.strip()]

        metrics = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty),
            "has_functions": bool(re.search(r'def \w+', code)),
            "has_classes": bool(re.search(r'class \w+', code)),
            "has_comments": bool(re.search(r'#\s', code)),
            "has_docstrings": bool(re.search(r'""".*?"""', code, re.DOTALL)),
            "has_imports": bool(re.search(r'import \w+', code)),
        }

        # Quality score
        score = 0.3  # Base
        if metrics["non_empty_lines"] > 3:
            score += 0.2
        if metrics["has_functions"] or metrics["has_classes"]:
            score += 0.2
        if metrics["has_comments"] or metrics["has_docstrings"]:
            score += 0.15
        if metrics["non_empty_lines"] > 10:
            score += 0.15

        metrics["quality_score"] = min(score, 1.0)
        return metrics

    def verify_example(self, example: Dict) -> Dict:
        """Verify a complete code training example."""
        results = {"type": "code_verification"}

        # Extract code from solution
        solution = example.get("solution", "")
        code_blocks = self.extract_code_blocks(solution)

        if not code_blocks:
            # Try the solution directly as code
            if solution.strip():
                code_blocks = [solution]

        results["num_code_blocks"] = len(code_blocks)

        # Verify each block
        block_results = []
        for i, block in enumerate(code_blocks):
            syntax = self.verify_python_syntax(block)
            quality = self.check_code_quality(block)
            block_results.append({
                "block_index": i,
                "syntax_valid": syntax["valid"],
                "syntax_error": syntax.get("error"),
                "quality": quality,
            })

        results["blocks"] = block_results
        results["all_syntax_valid"] = all(b["syntax_valid"] for b in block_results)
        results["overall_valid"] = results["all_syntax_valid"] and len(code_blocks) > 0

        return results
