"""
ALAN v4 — Curriculum-Based Training Schedule
ECONX GROUP (PTY) LTD

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

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Phase definitions
CURRICULUM_PHASES = {
    "foundation": {
        "name": "Foundation",
        "fraction": 0.30,
        "description": "Basic language understanding and simple reasoning",
        "data_categories": ["reasoning"],
        "difficulty_range": (1, 3),
        "learning_rate_multiplier": 1.0,
    },
    "capability": {
        "name": "Capability Building",
        "fraction": 0.30,
        "description": "Multi-step reasoning, code, conversation, topic tracking",
        "data_categories": ["reasoning", "conversation", "emotional"],
        "difficulty_range": (3, 6),
        "learning_rate_multiplier": 0.8,
    },
    "advanced": {
        "name": "Advanced Skills",
        "fraction": 0.25,
        "description": "Complex reasoning, wisdom, engagement, feedback",
        "data_categories": ["reasoning", "conversation", "emotional", "wisdom", "image_understanding"],
        "difficulty_range": (5, 10),
        "learning_rate_multiplier": 0.5,
    },
    "alignment": {
        "name": "Alignment & Polish",
        "fraction": 0.15,
        "description": "Safety, calibration, adversarial, personality",
        "data_categories": ["safety", "conversation", "emotional"],
        "difficulty_range": (1, 10),
        "learning_rate_multiplier": 0.3,
    },
}


class CurriculumScheduler:
    """
    Manages the training curriculum, controlling which data is used
    at each phase and how training parameters adjust over time.
    """

    def __init__(
        self,
        total_steps: int,
        data_dir: str = "data/generated",
        phases: Optional[Dict] = None,
    ):
        self.total_steps = total_steps
        self.data_dir = Path(data_dir)
        self.phases = phases or CURRICULUM_PHASES
        self.current_phase_idx = 0
        self.phase_names = list(self.phases.keys())

        # Compute step boundaries for each phase
        self.phase_boundaries = self._compute_boundaries()
        logger.info(f"[Curriculum] Initialized with {len(self.phases)} phases, {total_steps} total steps")
        for name, bounds in self.phase_boundaries.items():
            logger.info(f"  Phase '{name}': steps {bounds['start']} - {bounds['end']}")

    def _compute_boundaries(self) -> Dict[str, Dict]:
        """Compute start/end steps for each phase."""
        boundaries = {}
        current_step = 0
        for name, phase in self.phases.items():
            phase_steps = int(self.total_steps * phase["fraction"])
            boundaries[name] = {
                "start": current_step,
                "end": current_step + phase_steps,
                "steps": phase_steps,
            }
            current_step += phase_steps
        return boundaries

    def get_current_phase(self, step: int) -> str:
        """Get the current training phase for the given step."""
        for name, bounds in self.phase_boundaries.items():
            if bounds["start"] <= step < bounds["end"]:
                return name
        return self.phase_names[-1]  # Default to last phase

    def get_phase_config(self, step: int) -> Dict:
        """Get configuration for the current phase."""
        phase_name = self.get_current_phase(step)
        phase = self.phases[phase_name]
        bounds = self.phase_boundaries[phase_name]

        # Progress within current phase (0.0 to 1.0)
        phase_progress = (step - bounds["start"]) / max(bounds["steps"], 1)

        return {
            "phase_name": phase_name,
            "phase_display": phase["name"],
            "description": phase["description"],
            "data_categories": phase["data_categories"],
            "difficulty_range": phase["difficulty_range"],
            "lr_multiplier": phase["learning_rate_multiplier"],
            "phase_progress": phase_progress,
            "step": step,
        }

    def get_data_categories(self, step: int) -> List[str]:
        """Get which data categories to use at the current step."""
        phase_name = self.get_current_phase(step)
        return self.phases[phase_name]["data_categories"]

    def get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for current phase."""
        phase_name = self.get_current_phase(step)
        return self.phases[phase_name]["learning_rate_multiplier"]

    def filter_data_for_phase(
        self,
        examples: List[Dict],
        step: int,
    ) -> List[Dict]:
        """
        Filter training examples based on current curriculum phase.
        Returns only examples appropriate for the current phase.
        """
        config = self.get_phase_config(step)
        categories = config["data_categories"]
        diff_min, diff_max = config["difficulty_range"]

        filtered = []
        for ex in examples:
            ex_type = ex.get("type", "")
            ex_diff = ex.get("difficulty", 5)

            # Check category match
            category_match = any(cat in ex_type for cat in categories) or ex_type in categories

            # Check difficulty (if specified)
            if isinstance(ex_diff, str):
                diff_map = {"easy": 2, "medium": 5, "hard": 7, "expert": 9}
                ex_diff = diff_map.get(ex_diff, 5)

            diff_match = diff_min <= ex_diff <= diff_max

            if category_match and diff_match:
                filtered.append(ex)

        # If no matches, return all (fallback)
        return filtered if filtered else examples

    def log_phase_transition(self, step: int, prev_phase: str, new_phase: str):
        """Log a phase transition."""
        logger.info(f"[Curriculum] Phase transition at step {step}: {prev_phase} → {new_phase}")

    def get_summary(self) -> Dict:
        """Get a summary of the curriculum schedule."""
        return {
            "total_steps": self.total_steps,
            "num_phases": len(self.phases),
            "phases": {
                name: {
                    "fraction": phase["fraction"],
                    "steps": self.phase_boundaries[name]["steps"],
                    "categories": phase["data_categories"],
                }
                for name, phase in self.phases.items()
            },
        }
