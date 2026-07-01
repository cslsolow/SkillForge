"""Go language adapter for the synthesis pipeline.

Test identifier format: '<relative_package_dir>::<TestFunctionName>'
  e.g.  './pkg/parser::TestParse'
        './cmd/server::TestHandleRequest'

Go coverage is collected via 'go test -coverprofile' and the .out file
is parsed to extract executed line ranges.
"""
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import LanguageAdapter


_TEST_FUNC_RE = re.compile(r"^func\s+(Test\w+)\s*\(", re.MULTILINE)
_COVER_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<sl>\d+)\.\d+,(?P<el>\d+)\.\d+\s+\d+\s+(?P<cnt>\d+)$"
)


class GoAdapter(LanguageAdapter):
    """Adapter for Go repositories using 'go test' and go coverage."""

    @property
    def name(self) -> str:
        return "go"

    @property
    def code_fence(self) -> str:
        return "go"

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def setup_environment(
        self, instance_id: str, instance_data: Dict, repos_dir: Path
    ) -> bool:
        """For Go repos we just need to run 'go mod download'."""
        repo_path = repos_dir / instance_id / "repo"
        if not repo_path.exists():
            return False
        go_mod = repo_path / "go.mod"
        if not go_mod.exists():
            print(f"[GoAdapter] No go.mod found in {repo_path}")
            return False
        try:
            result = subprocess.run(
                ["go", "mod", "download"],
                cwd=repo_path, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"[GoAdapter] go mod download failed: {result.stderr[:300]}")
                return False
            return True
        except Exception as e:
            print(f"[GoAdapter] setup failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Test discovery
    # ------------------------------------------------------------------

    def discover_tests(
        self, instance_id: str, repos_dir: Path, timeout: int
    ) -> List[str]:
        """Discover Go tests by grepping *_test.go files for func Test*."""
        repo_path = repos_dir / instance_id / "repo"
        if not repo_path.exists():
            return []

        tests: List[str] = []
        seen = set()
        for test_file in repo_path.rglob("*_test.go"):
            try:
                content = test_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel_file = test_file.relative_to(repo_path)
            pkg_dir = "./" + str(rel_file.parent) if str(rel_file.parent) != "." else "."
            for m in _TEST_FUNC_RE.finditer(content):
                func_name = m.group(1)
                test_id = f"{pkg_dir}::{func_name}"
                if test_id not in seen:
                    seen.add(test_id)
                    tests.append(test_id)
        return tests

    # ------------------------------------------------------------------
    # Test source
    # ------------------------------------------------------------------

    def get_test_source(self, test_str: str, repo_path: Path) -> Optional[str]:
        """Extract the test function body from the *_test.go files."""
        pkg_dir, func_name = _split_test_str(test_str)
        pkg_path = repo_path / pkg_dir.lstrip("./")

        for test_file in pkg_path.glob("*_test.go"):
            try:
                content = test_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            src = _extract_go_func(content, func_name)
            if src:
                return src
        return None

    # ------------------------------------------------------------------
    # Coverage tracing
    # ------------------------------------------------------------------

    def collect_coverage(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Dict:
        pkg_dir, func_name = _split_test_str(test_str)
        module_name = _get_go_module(repo_path)

        with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as tf:
            cover_file = tf.name

        cmd = [
            "go", "test",
            "-run", f"^{func_name}$",
            "-coverprofile", cover_file,
            "-coverpkg", "./...",
            "-v",
            pkg_dir,
        ]
        try:
            subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"[GoAdapter] coverage run failed: {e}")
            return {"total_files": 0, "file_coverage": {}}

        trace_summary = _parse_go_cover(cover_file, repo_path, module_name)
        try:
            Path(cover_file).unlink()
        except Exception:
            pass
        return trace_summary

    # ------------------------------------------------------------------
    # Syntax checking
    # ------------------------------------------------------------------

    def check_syntax(self, file_path: str, content: str) -> Optional[str]:
        """Use 'gofmt -e' to validate Go syntax."""
        try:
            result = subprocess.run(
                ["gofmt", "-e", file_path],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return result.stderr or result.stdout
            return None
        except FileNotFoundError:
            # gofmt not available — skip syntax check
            return None
        except Exception as e:
            return str(e)

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def run_test(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Tuple[int, str]:
        cmd = self.make_test_command(test_str, repo_path)
        result = subprocess.run(
            cmd, shell=True, cwd=repo_path,
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr

    def make_test_command(self, test_str: str, repo_path: Path) -> str:
        pkg_dir, func_name = _split_test_str(test_str)
        return f"go test -run '^{func_name}$' -v {pkg_dir}"

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def get_developer_system_prompt(
        self, base_indent: int, is_complete_function: bool, extra_context: str, indent_str: str
    ) -> str:
        if is_complete_function:
            return (
                "You are a Go developer. You need to implement a complete function/method.\n\n"
                "Key rules:\n"
                "1. You cannot see the original implementation — infer purpose from context.\n"
                f"2. Base indentation is {base_indent} spaces (Go typically uses tabs; adjust if needed).\n"
                "3. Code must be syntactically correct, compilable, and semantically complete.\n"
                f"4. {extra_context}\n\n"
                "Implement the full function body. Do not use placeholder comments or panic()."
            )
        else:
            return (
                "You are a Go developer. Fill in the [MASKED] code segment from context.\n\n"
                "Key rules:\n"
                "1. You cannot see the original code — infer it from context only.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct, compilable, and semantically complete.\n"
                "4. Do not fill with panic() or empty blocks.\n\n"
                f"Indentation example:\n"
                f"{indent_str}firstLine()\n"
                f"{indent_str}\tdeeper indentation if needed\n"
                f"{indent_str}return result"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_test_str(test_str: str) -> Tuple[str, str]:
    """Split '<pkg_dir>::<TestName>' into (pkg_dir, func_name)."""
    if "::" in test_str:
        pkg_dir, func_name = test_str.split("::", 1)
    else:
        pkg_dir, func_name = ".", test_str
    return pkg_dir or ".", func_name


def _get_go_module(repo_path: Path) -> str:
    """Read the module name from go.mod."""
    go_mod = repo_path / "go.mod"
    if not go_mod.exists():
        return ""
    for line in go_mod.read_text().splitlines():
        line = line.strip()
        if line.startswith("module "):
            return line.split()[1]
    return ""


def _extract_go_func(content: str, func_name: str) -> Optional[str]:
    """Extract a top-level Go function body from source text."""
    pattern = re.compile(
        rf"^func\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{", re.MULTILINE
    )
    m = pattern.search(content)
    if not m:
        return None
    start = m.start()
    # Walk forward to find the matching closing brace
    depth = 0
    i = m.end() - 1  # position of opening brace
    while i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
        i += 1
    return content[start:]


def _parse_go_cover(cover_file: str, repo_path: Path, module_name: str) -> Dict:
    """Parse a Go coverage profile into the standard trace_summary format."""
    trace_summary: Dict = {"total_files": 0, "file_coverage": {}}
    cover_path = Path(cover_file)
    if not cover_path.exists():
        return trace_summary

    # Build a mapping: relative_path -> set of executed line numbers
    file_lines_executed: Dict[str, set] = {}
    try:
        for line in cover_path.read_text().splitlines():
            m = _COVER_LINE_RE.match(line)
            if not m:
                continue
            if int(m.group("cnt")) == 0:
                continue
            filepath = m.group("file")
            # Strip module prefix to get relative path
            if module_name and filepath.startswith(module_name + "/"):
                rel_path = filepath[len(module_name) + 1:]
            else:
                # Try to strip common prefixes
                rel_path = filepath
            start_line = int(m.group("sl"))
            end_line = int(m.group("el"))
            file_lines_executed.setdefault(rel_path, set()).update(
                range(start_line, end_line + 1)
            )
    except Exception as e:
        print(f"[GoAdapter] cover parse error: {e}")
        return trace_summary

    for rel_path, lines in file_lines_executed.items():
        # Skip test files
        if rel_path.endswith("_test.go"):
            continue
        abs_path = repo_path / rel_path
        if not abs_path.exists():
            continue
        executed_code: Dict[str, str] = {}
        try:
            file_content = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in sorted(lines):
                if 0 < ln <= len(file_content):
                    executed_code[str(ln)] = file_content[ln - 1].rstrip()
        except Exception:
            pass
        trace_summary["file_coverage"][rel_path] = {
            "absolute_path": str(abs_path),
            "executed_lines": sorted(lines),
            "total_executed_lines": len(lines),
            "executed_code": executed_code,
        }

    trace_summary["total_files"] = len(trace_summary["file_coverage"])
    return trace_summary
