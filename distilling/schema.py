"""
Data classes shared across the SkillForge pipeline.

Trajectory types (TrajectoryStep, Trajectory, GoldenInfo) represent the
input from an SWE-agent run. Extraction types (CodeScope, CodeAccess,
KeypointItem, EnvKnowledgeItem) represent the extracted skill records.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Trajectory types
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    index: int
    thought: str
    action: str
    observation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    info: Dict[str, Any]
    steps: List[TrajectoryStep]
    instance_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "info": self.info,
            "instance_id": self.instance_id,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class GoldenInfo:
    instance_id: str
    patch: str
    test_patch: str
    problem_statement: str
    raw: Dict[str, Any]
    solved: Optional[bool] = None
    modified_files: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "problem_statement": self.problem_statement,
            "solved": self.solved,
            "modified_files": self.modified_files,
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# Code-access / scope types (used by the extractor)
# ---------------------------------------------------------------------------


class AccessType(str, Enum):
    SEARCH = "search"
    VIEW = "view"
    EDIT = "edit"


@dataclass(frozen=True)
class CodeScope:
    filepath: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None

    @property
    def scope_key(self) -> str:
        parts = [self.filepath]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return "::".join(parts)

    @property
    def scope_type(self) -> str:
        if self.function_name:
            return "function"
        if self.class_name:
            return "class"
        return "file"


@dataclass(frozen=True)
class CodeAccess:
    step_index: int
    filepath: str
    access_type: AccessType
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    thought: str = ""
    action: str = ""


# ---------------------------------------------------------------------------
# Extraction output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeypointItem:
    api_path: str
    experience_or_reflection: str
    issue: str
    thinking: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_path": self.api_path,
            "experience_or_reflection": self.experience_or_reflection,
            "issue": self.issue,
            "thinking": self.thinking,
        }


@dataclass(frozen=True)
class EnvKnowledgeItem:
    api_path: str
    purpose: str
    playbook: str = ""
    related_apis: List[Dict[str, str]] = field(default_factory=list)
    thinking: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_path": self.api_path,
            "purpose": self.purpose,
            "playbook": self.playbook,
            "related_apis": list(self.related_apis),
            "thinking": self.thinking,
        }
