"""Shared source-file helpers for distilling and runtime skill matching."""

from __future__ import annotations

SOURCE_EXTENSIONS = (".tsx", ".jsx", ".py", ".go", ".ts", ".js")
TEST_FILE_MARKERS = (
    "/tests/",
    "/test/",
    "__tests__/",
    ".test.",
    ".spec.",
    "_test.go",
)


def is_source_file(filepath: str) -> bool:
    return filepath.endswith(SOURCE_EXTENSIONS)


def is_test_file(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/").lstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    if name.startswith("test_") or name.endswith("_test.py") or name.endswith("_test.go"):
        return True
    marked = f"/{normalized}"
    return any(marker in marked for marker in TEST_FILE_MARKERS)
