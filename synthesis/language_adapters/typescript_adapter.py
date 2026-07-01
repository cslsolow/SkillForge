"""TypeScript language adapter for the synthesis pipeline.

Test identifier format: '<relative_test_file>::<test_name>'
  e.g.  'src/parser.test.ts::should parse empty input'
        '__tests__/utils.spec.ts::Utils::format date correctly'

Coverage is collected via Jest's --coverage --coverageReporters=json flag.
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import LanguageAdapter


class TypeScriptAdapter(LanguageAdapter):
    """Adapter for TypeScript repositories using Jest."""

    # Patterns for test file discovery
    _TEST_FILE_PATTERNS = ("*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx",
                           "*.test.js", "*.spec.js")
    # Patterns for extracting test names from source
    _TEST_NAME_RE = re.compile(
        r"""(?:it|test|describe)\s*\(\s*['"`](?P<name>[^'"`]+)['"`]""",
        re.MULTILINE,
    )

    @property
    def name(self) -> str:
        return "typescript"

    @property
    def code_fence(self) -> str:
        return "typescript"

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def setup_environment(
        self, instance_id: str, instance_data: Dict, repos_dir: Path
    ) -> bool:
        """Run npm install (or yarn install) inside the cloned repo."""
        repo_path = repos_dir / instance_id / "repo"
        if not repo_path.exists():
            return False
        pkg_json = repo_path / "package.json"
        if not pkg_json.exists():
            print(f"[TypeScriptAdapter] No package.json found in {repo_path}")
            return False
        try:
            # Prefer yarn if yarn.lock exists
            if (repo_path / "yarn.lock").exists():
                cmd = ["yarn", "install", "--frozen-lockfile"]
            else:
                cmd = ["npm", "install", "--legacy-peer-deps"]
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                print(f"[TypeScriptAdapter] install failed: {result.stderr[:300]}")
                return False
            return True
        except Exception as e:
            print(f"[TypeScriptAdapter] setup failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Test discovery
    # ------------------------------------------------------------------

    def discover_tests(
        self, instance_id: str, repos_dir: Path, timeout: int
    ) -> List[str]:
        """Find test files and extract test names via regex."""
        repo_path = repos_dir / instance_id / "repo"
        if not repo_path.exists():
            return []

        tests: List[str] = []
        seen = set()

        for pattern in self._TEST_FILE_PATTERNS:
            for test_file in repo_path.rglob(pattern):
                # Skip node_modules
                if "node_modules" in test_file.parts:
                    continue
                try:
                    content = test_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                rel_file = str(test_file.relative_to(repo_path))
                for m in self._TEST_NAME_RE.finditer(content):
                    test_name = m.group("name").strip()
                    test_id = f"{rel_file}::{test_name}"
                    if test_id not in seen:
                        seen.add(test_id)
                        tests.append(test_id)
        return tests

    # ------------------------------------------------------------------
    # Test source
    # ------------------------------------------------------------------

    def get_test_source(self, test_str: str, repo_path: Path) -> Optional[str]:
        """Return the content around the test function in the test file."""
        test_file, test_name = _split_test_str(test_str)
        abs_path = repo_path / test_file
        if not abs_path.exists():
            return None
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        # Try to extract just the block matching test_name
        block = _extract_ts_test_block(content, test_name)
        return block if block else content[:3000]  # fallback: first 3000 chars

    # ------------------------------------------------------------------
    # Coverage tracing
    # ------------------------------------------------------------------

    def collect_coverage(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Dict:
        test_file, test_name = _split_test_str(test_str)

        cov_dir = repo_path / ".jest_coverage_tmp"
        cov_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._jest_cmd(repo_path) + [
            "--testPathPattern", re.escape(test_file.replace("\\", "/")),
            "--testNamePattern", re.escape(test_name),
            "--coverage",
            "--coverageReporters", "json",
            "--coverageDirectory", str(cov_dir),
            "--no-cache",
            "--forceExit",
        ]
        env = os.environ.copy()
        env["CI"] = "true"  # suppress interactive prompts
        try:
            subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True,
                timeout=timeout, env=env
            )
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"[TypeScriptAdapter] coverage run failed: {e}")
            return {"total_files": 0, "file_coverage": {}}

        coverage_json = cov_dir / "coverage-final.json"
        trace_summary = _parse_jest_coverage(coverage_json, repo_path)

        # Clean up
        try:
            import shutil
            shutil.rmtree(cov_dir, ignore_errors=True)
        except Exception:
            pass

        return trace_summary

    # ------------------------------------------------------------------
    # Syntax checking
    # ------------------------------------------------------------------

    def check_syntax(self, file_path: str, content: str) -> Optional[str]:
        """Detect syntax errors by examining TypeScript/JS error patterns.

        We rely on a lightweight check: if the file has obviously unbalanced
        braces or the test run fails with a SyntaxError, it will be detected.
        For a quick pre-check we attempt to run 'node --check' on transpiled
        content if ts-node is available, otherwise fall back to brace counting.
        """
        # Quick brace balance check
        open_b = content.count("{") - content.count("}")
        open_p = content.count("(") - content.count(")")
        open_sq = content.count("[") - content.count("]")
        if abs(open_b) > 2 or abs(open_p) > 2 or abs(open_sq) > 2:
            return f"Unbalanced delimiters: {{={open_b:+d} (={open_p:+d} [={open_sq:+d}"

        # Try tsc --noEmit if tsconfig is available in the parent dir
        try:
            fp = Path(file_path)
            tsconfig = _find_tsconfig(fp.parent)
            if tsconfig and shutil_which("tsc"):
                result = subprocess.run(
                    ["tsc", "--noEmit", "--skipLibCheck", "--project", str(tsconfig)],
                    cwd=fp.parent, capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    err_lines = result.stdout.splitlines() + result.stderr.splitlines()
                    syntax_errs = [l for l in err_lines if "error TS" in l and str(fp.name) in l]
                    if syntax_errs:
                        return "\n".join(syntax_errs[:5])
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def run_test(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Tuple[int, str]:
        cmd = self._build_run_cmd(test_str, repo_path)
        env = os.environ.copy()
        env["CI"] = "true"
        result = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True,
            timeout=timeout, env=env
        )
        return result.returncode, result.stdout + result.stderr

    def make_test_command(self, test_str: str, repo_path: Path) -> str:
        cmd_list = self._build_run_cmd(test_str, repo_path)
        return " ".join(cmd_list)

    def _build_run_cmd(self, test_str: str, repo_path: Path) -> List[str]:
        test_file, test_name = _split_test_str(test_str)
        return self._jest_cmd(repo_path) + [
            "--testPathPattern", re.escape(test_file.replace("\\", "/")),
            "--testNamePattern", re.escape(test_name),
            "--no-coverage",
            "--forceExit",
        ]

    def _jest_cmd(self, repo_path: Path) -> List[str]:
        """Return the base jest invocation for this repo."""
        # Prefer local jest binary
        local_jest = repo_path / "node_modules" / ".bin" / "jest"
        if local_jest.exists():
            return [str(local_jest)]
        return ["npx", "--yes", "jest"]

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def get_developer_system_prompt(
        self, base_indent: int, is_complete_function: bool, extra_context: str, indent_str: str
    ) -> str:
        if is_complete_function:
            return (
                "You are a TypeScript developer. Implement a complete function/method.\n\n"
                "Key rules:\n"
                "1. You cannot see the original implementation — infer purpose from context.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct, type-safe, and semantically complete.\n"
                f"4. {extra_context}\n\n"
                "Implement the full function body. Use proper TypeScript types. "
                "Do not use placeholder comments or throw new Error('not implemented')."
            )
        else:
            return (
                "You are a TypeScript developer. Fill in the [MASKED] code segment from context.\n\n"
                "Key rules:\n"
                "1. You cannot see the original code — infer it from context only.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct, type-safe, and semantically complete.\n"
                "4. Do not fill with placeholder statements.\n\n"
                f"Indentation example:\n"
                f"{indent_str}const x = value;\n"
                f"{indent_str}    // deeper indentation if needed\n"
                f"{indent_str}return result;"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_test_str(test_str: str) -> Tuple[str, str]:
    """Split '<test_file>::<test_name>' into (test_file, test_name)."""
    if "::" in test_str:
        idx = test_str.index("::")
        return test_str[:idx], test_str[idx + 2:]
    return test_str, ""


def _extract_ts_test_block(content: str, test_name: str) -> Optional[str]:
    """Extract the source block of the test with the given name."""
    # Escape special regex chars in test name
    escaped = re.escape(test_name)
    pattern = re.compile(
        rf"""(?:it|test)\s*\(\s*['"`]{escaped}['"`]""", re.MULTILINE
    )
    m = pattern.search(content)
    if not m:
        return None
    start = m.start()
    # Find opening brace of the callback
    brace_pos = content.find("{", m.end())
    if brace_pos == -1:
        return content[start:start + 500]
    depth = 0
    i = brace_pos
    while i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
        i += 1
    return content[start:]


def _find_tsconfig(start_dir: Path) -> Optional[Path]:
    """Walk upward looking for tsconfig.json."""
    current = start_dir.resolve()
    for _ in range(6):
        tsconfig = current / "tsconfig.json"
        if tsconfig.exists():
            return tsconfig
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def shutil_which(name: str) -> Optional[str]:
    import shutil
    return shutil.which(name)


def _parse_jest_coverage(coverage_json: Path, repo_path: Path) -> Dict:
    """Parse a Jest coverage-final.json into the standard trace_summary format."""
    trace_summary: Dict = {"total_files": 0, "file_coverage": {}}
    if not coverage_json.exists():
        return trace_summary

    try:
        data = json.loads(coverage_json.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[TypeScriptAdapter] coverage json parse error: {e}")
        return trace_summary

    for abs_path_str, file_data in data.items():
        abs_path = Path(abs_path_str)
        # Skip test files
        if any(pat in abs_path.name for pat in (".test.", ".spec.")):
            continue
        if "node_modules" in abs_path.parts:
            continue
        try:
            rel_path = str(abs_path.relative_to(repo_path))
        except ValueError:
            continue

        statement_map = file_data.get("statementMap", {})
        statement_counts = file_data.get("s", {})

        executed_lines: set = set()
        for stmt_id, count in statement_counts.items():
            if int(count) > 0 and stmt_id in statement_map:
                loc = statement_map[stmt_id]
                sl = loc.get("start", {}).get("line", 0)
                el = loc.get("end", {}).get("line", sl)
                if sl > 0:
                    executed_lines.update(range(sl, el + 1))

        if not executed_lines:
            continue

        executed_code: Dict[str, str] = {}
        try:
            file_lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in sorted(executed_lines):
                if 0 < ln <= len(file_lines):
                    executed_code[str(ln)] = file_lines[ln - 1].rstrip()
        except Exception:
            pass

        trace_summary["file_coverage"][rel_path] = {
            "absolute_path": abs_path_str,
            "executed_lines": sorted(executed_lines),
            "total_executed_lines": len(executed_lines),
            "executed_code": executed_code,
        }

    trace_summary["total_files"] = len(trace_summary["file_coverage"])
    return trace_summary
