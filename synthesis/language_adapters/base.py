"""Abstract base class for language-specific pipeline adapters."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LanguageAdapter(ABC):
    """Encapsulates language-specific operations for the synthesis pipeline.

    Each concrete adapter must implement all abstract methods to support
    test discovery, coverage tracing, syntax validation, and test execution
    for a specific programming language.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Language name, e.g. 'python', 'go', 'typescript'."""
        ...

    @property
    @abstractmethod
    def code_fence(self) -> str:
        """Markdown code-fence label, e.g. 'python', 'go', 'typescript'."""
        ...

    @abstractmethod
    def setup_environment(
        self, instance_id: str, instance_data: Dict, repos_dir: Path
    ) -> bool:
        """Set up the language runtime environment for a cloned repo.

        Args:
            instance_id: SWE-bench instance identifier.
            instance_data: Metadata dict from the dataset.
            repos_dir: Root directory where instances are stored.

        Returns:
            True on success, False on failure.
        """
        ...

    @abstractmethod
    def discover_tests(
        self, instance_id: str, repos_dir: Path, timeout: int
    ) -> List[str]:
        """Return test identifiers for the given repo instance.

        Format (language-specific):
          Python:     'tests/test_foo.py::TestClass::test_method'
          Go:         './pkg/parser::TestParse'
          TypeScript: 'src/parser.test.ts::should parse empty input'
        """
        ...

    @abstractmethod
    def get_test_source(self, test_str: str, repo_path: Path) -> Optional[str]:
        """Return the source code of the test function/block."""
        ...

    @abstractmethod
    def collect_coverage(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Dict:
        """Run the test with coverage instrumentation and return a trace_summary dict.

        The returned dict MUST have this structure (compatible with the existing
        _extract_segments_with_context pipeline):

            {
                'file_coverage': {
                    'rel/path/file.ext': {
                        'absolute_path': '/abs/path/file.ext',
                        'executed_lines': [1, 2, 3, ...],
                        'total_executed_lines': N,
                        'executed_code': {
                            '1': 'line 1 content',
                            '2': 'line 2 content',
                            ...
                        }
                    },
                    ...
                },
                'total_files': N
            }

        Only non-test source files should be included (exclude *_test.go,
        *.test.ts, *.spec.ts, tests/, etc.).
        """
        ...

    @abstractmethod
    def check_syntax(self, file_path: str, content: str) -> Optional[str]:
        """Check whether *content* is syntactically valid for the language.

        Args:
            file_path: Absolute path where content will be (or already is) written.
            content: Source code to validate.

        Returns:
            None if valid, or an error message string if invalid.
        """
        ...

    @abstractmethod
    def run_test(
        self, test_str: str, repo_path: Path, timeout: int
    ) -> Tuple[int, str]:
        """Run a single test and return (returncode, combined_stdout_stderr)."""
        ...

    @abstractmethod
    def make_test_command(self, test_str: str, repo_path: Path) -> str:
        """Build the shell command string used to run the test.

        This is stored in the instance metadata for reproducibility.
        """
        ...

    def get_developer_system_prompt(
        self,
        base_indent: int,
        is_complete_function: bool,
        extra_context: str,
        indent_str: str,
    ) -> str:
        """Return the system prompt for the code-generation LLM call.

        Subclasses may override to add language-specific rules.
        """
        lang = self.name.capitalize()
        if is_complete_function:
            return (
                f"You are a {lang} developer. Implement a complete function/method.\n\n"
                "Key rules:\n"
                "1. You cannot see the original implementation — infer its purpose from context.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct and semantically complete.\n"
                f"4. {extra_context}\n\n"
                "Implement the function body. Do not use placeholder comments or pass statements."
            )
        else:
            return (
                f"You are a {lang} developer. Fill in the [MASKED] code segment from context.\n\n"
                "Key rules:\n"
                "1. You cannot see the original code — infer it from context only.\n"
                f"2. Base indentation is {base_indent} spaces.\n"
                "3. Code must be syntactically correct and semantically complete.\n"
                "4. Do not use placeholder statements.\n\n"
                f"Indentation example:\n"
                f"{indent_str}first line of code\n"
                f"{indent_str}    deeper indentation (+4 spaces) if needed\n"
                f"{indent_str}second line of code"
            )
