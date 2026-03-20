"""AST-based Python file scope index: line-range → (class, function)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


def _get_docstring(node: ast.AST) -> str:
    try:
        return ast.get_docstring(node) or ""
    except Exception:
        return ""


@dataclass
class FunctionInfo:
    name: str
    start_line: int
    end_line: int
    docstring: str = ""
    class_name: str | None = None


@dataclass
class ClassInfo:
    name: str
    start_line: int
    end_line: int
    docstring: str = ""
    methods: dict[str, FunctionInfo] = field(default_factory=dict)


@dataclass
class FileStructure:
    filepath: str
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)


def parse_file_structure(filepath: str, content: str) -> FileStructure:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return FileStructure(filepath=filepath)

    classes: dict[str, ClassInfo] = {}
    functions: dict[str, FunctionInfo] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_methods: dict[str, FunctionInfo] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_methods[item.name] = FunctionInfo(
                        name=item.name,
                        start_line=item.lineno,
                        end_line=item.end_lineno or item.lineno,
                        docstring=_get_docstring(item),
                        class_name=node.name,
                    )
            classes[node.name] = ClassInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                docstring=_get_docstring(node),
                methods=class_methods,
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = FunctionInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                docstring=_get_docstring(node),
                class_name=None,
            )

    return FileStructure(filepath=filepath, classes=classes, functions=functions)


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


class CodeIndex:
    def __init__(self) -> None:
        self._file_structures: dict[str, FileStructure] = {}

    def add_file(self, filepath: str, content: str) -> FileStructure:
        structure = parse_file_structure(filepath, content)
        self._file_structures[filepath] = structure
        return structure

    def add_file_from_repo(self, repo_root: Path, filepath: str) -> FileStructure:
        full_path = repo_root / filepath
        if not full_path.exists() or full_path.suffix != ".py":
            structure = FileStructure(filepath=filepath)
            self._file_structures[filepath] = structure
            return structure
        return self.add_file(
            filepath,
            full_path.read_text(encoding="utf-8", errors="replace"),
        )

    def find_scope_for_range(
        self,
        filepath: str,
        start_line: int | None,
        end_line: int | None,
    ) -> list[tuple[str | None, str | None]]:
        structure = self._file_structures.get(filepath)
        if not structure:
            return [(None, None)]
        if start_line is None or end_line is None:
            return [(None, None)]

        scopes: list[tuple[str | None, str | None]] = []
        seen: set[tuple[str | None, str | None]] = set()

        for class_info in structure.classes.values():
            if _ranges_overlap(start_line, end_line, class_info.start_line, class_info.end_line):
                for method_info in class_info.methods.values():
                    if _ranges_overlap(start_line, end_line, method_info.start_line, method_info.end_line):
                        key = (class_info.name, method_info.name)
                        if key not in seen:
                            scopes.append(key)
                            seen.add(key)
                if not any(s[0] == class_info.name and s[1] is not None for s in scopes):
                    key = (class_info.name, None)
                    if key not in seen:
                        scopes.append(key)
                        seen.add(key)

        for func_info in structure.functions.values():
            if _ranges_overlap(start_line, end_line, func_info.start_line, func_info.end_line):
                key = (None, func_info.name)
                if key not in seen:
                    scopes.append(key)
                    seen.add(key)

        if not scopes:
            scopes.append((None, None))
        return scopes
