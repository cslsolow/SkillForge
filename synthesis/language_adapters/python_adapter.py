"""Python language adapter for the synthesis pipeline."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import LanguageAdapter


class PythonAdapter(LanguageAdapter):
    """Adapter for Python repositories using pytest and coverage.py."""

    def __init__(self, python_path: Optional[str] = None):
        self._python_path = python_path  # can be set later per-repo

    @property
    def name(self) -> str:
        return "python"

    @property
    def code_fence(self) -> str:
        return "python"

    def set_python_path(self, python_path: str):
        """Set the venv python path (called by setup_repos or the generator)."""
        self._python_path = python_path

    def _py(self, repo_path: Path) -> str:
        """Return the Python executable to use for this repo."""
        if self._python_path and Path(self._python_path).exists():
            return self._python_path
        # fallback: venv inside repo or system python
        venv_py = repo_path / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable

    def setup_environment(
        self, instance_id: str, instance_data: Dict, repos_dir: Path
    ) -> bool:
        repo_path = repos_dir / instance_id / "repo"
        venv_path = repos_dir / instance_id / "venv"
        if venv_path.exists():
            return True
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True, capture_output=True)
            pip = venv_path / "bin" / "pip"
            subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True, capture_output=True, timeout=300)
            subprocess.run([str(pip), "install", "setuptools", "wheel"], check=True, capture_output=True, timeout=300)
            subprocess.run([str(pip), "install", "-e", str(repo_path)], check=True, capture_output=True, timeout=900)
            subprocess.run([str(pip), "install", "pytest", "pytest-timeout", "coverage"], check=True, capture_output=True, timeout=300)
            return True
        except Exception as e:
            print(f"Python env setup failed for {instance_id}: {e}")
            return False

    def discover_tests(
        self, instance_id: str, repos_dir: Path, timeout: int
    ) -> List[str]:
        repo_path = repos_dir / instance_id / "repo"
        venv_python = repos_dir / instance_id / "venv" / "bin" / "python"
        if not repo_path.exists() or not venv_python.exists():
            return []
        cmd = [
            str(venv_python), "-m", "pytest",
            "--collect-only", "-q",
            "--rootdir", str(repo_path),
            "--ignore=venv",
        ]
        try:
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout)
        except Exception:
            return []
        tests, seen = [], set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or "::" not in line:
                continue
            node_id = line.split("[")[0] if "[" in line else line
            if node_id in seen or not re.match(r".*\.py::", node_id):
                continue
            seen.add(node_id)
            tests.append(node_id)
        return tests

    def get_test_source(self, test_str: str, repo_path: Path) -> Optional[str]:
        parts = test_str.split("::")
        file_path = parts[0]
        module_name = file_path.removesuffix(".py").replace("/", ".")
        test_class = parts[1] if len(parts) == 3 else ""
        test_method = parts[-1]

        if test_class:
            import_block = (
                f"test_class_obj = getattr(module, '{test_class}')\n"
                f"test_instance = test_class_obj()\n"
                f"method = getattr(test_instance, '{test_method}')"
            )
        else:
            import_block = f"method = getattr(module, '{test_method}')"

        script = (
            f"import sys, inspect\n"
            f"sys.path.insert(0, '{repo_path}')\n"
            f"module = __import__('{module_name}', fromlist=[''])\n"
            f"{import_block}\n"
            f"print(inspect.getsource(method))"
        )
        py = self._py(repo_path)
        result = subprocess.run([py, "-c", script], cwd=repo_path, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return result.stdout

    def collect_coverage(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Dict:
        parts = test_str.split("::")
        file_path = parts[0]
        test_class = parts[1] if len(parts) == 3 else ""
        test_method = parts[-1]
        if test_class:
            test_path = f"{file_path}::{test_class}::{test_method}"
        else:
            test_path = f"{file_path}::{test_method}"

        py = self._py(repo_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_path}:{env.get('PYTHONPATH', '')}"

        subprocess.run(
            [py, "-m", "coverage", "run", "--source", str(repo_path), "-m", "pytest", test_path, "-xvs"],
            cwd=repo_path, capture_output=True, text=True, timeout=timeout, env=env
        )

        coverage_json = repo_path / ".coverage_report.json"
        subprocess.run(
            [py, "-m", "coverage", "json", "-o", str(coverage_json)],
            cwd=repo_path, capture_output=True, timeout=60
        )

        trace_summary: Dict = {"total_files": 0, "file_coverage": {}}
        if coverage_json.exists():
            try:
                data = json.loads(coverage_json.read_text())
                for filepath, file_data in data.get("files", {}).items():
                    rel_path = str(Path(filepath).relative_to(repo_path))
                    # Exclude test files
                    if any(p in rel_path for p in ("test_", "tests/", "_test.py")):
                        continue
                    executed_lines = file_data.get("executed_lines", [])
                    if not executed_lines:
                        continue
                    executed_code: Dict[str, str] = {}
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            file_lines = f.readlines()
                        for ln in executed_lines:
                            if 0 < ln <= len(file_lines):
                                executed_code[str(ln)] = file_lines[ln - 1].rstrip()
                    except Exception:
                        pass
                    trace_summary["file_coverage"][rel_path] = {
                        "absolute_path": filepath,
                        "executed_lines": executed_lines,
                        "total_executed_lines": len(executed_lines),
                        "executed_code": executed_code,
                    }
                coverage_json.unlink(missing_ok=True)
            except Exception as e:
                print(f"Coverage parse failed: {e}")
        trace_summary["total_files"] = len(trace_summary["file_coverage"])
        return trace_summary

    def check_syntax(self, file_path: str, content: str) -> Optional[str]:
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return str(e)

    def run_test(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Tuple[int, str]:
        cmd = self.make_test_command(test_str, repo_path)
        result = subprocess.run(cmd, shell=True, cwd=repo_path, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr

    def make_test_command(self, test_str: str, repo_path: Path) -> str:
        py = self._py(repo_path)
        return f"{py} -m pytest {test_str} -x"

    def get_developer_system_prompt(
        self, base_indent: int, is_complete_function: bool, extra_context: str, indent_str: str
    ) -> str:
        if is_complete_function:
            return (
                f"You are a Python developer. You need to implement a complete function/method.\n\n"
                "Key rules:\n"
                "1. You cannot see the original implementation — infer its purpose from context.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct, runnable, and semantically complete.\n"
                f"4. {extra_context}\n\n"
                "For this complete function/method, you must:\n"
                "- Understand the function's intent (infer from name, parameters, and context).\n"
                "- Implement the core functional logic.\n"
                "- The generated code must be a semantically complete block — do not use pass.\n\n"
                f"Indentation example:\n"
                f"{indent_str}def function_name(...):\n"
                f"{indent_str}    # function body (+4 spaces)\n"
                f"{indent_str}    return result"
            )
        else:
            return (
                "You are a Python developer. You need to fill in the code marked as [MASKED] based on context.\n\n"
                "Key rules:\n"
                "1. You cannot see the original code — infer it from context only.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct, runnable, and semantically complete.\n"
                "4. The generated code must be a natural code block — do not fill with multiple pass statements.\n\n"
                f"Indentation example:\n"
                f"{indent_str}first line of code\n"
                f"{indent_str}    deeper indentation (+4 spaces) if needed\n"
                f"{indent_str}second line of code"
            )
