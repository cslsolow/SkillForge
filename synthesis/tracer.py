"""Trace all code paths executed during test case runs using sys.settrace."""
import sys
import os
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json
from pathlib import Path


class CodeTracer:
    """Trace code accessed during test execution."""

    def __init__(self, repo_path: str, exclude_patterns: List[str] = None, max_depth: int = None):
        """
        Args:
            repo_path: Root path of the repository.
            exclude_patterns: Path patterns to exclude (e.g. 'tests/', 'venv/').
            max_depth: Maximum call-stack depth limit (to filter deep dependencies).
        """
        self.repo_path = Path(repo_path).resolve()
        self.exclude_patterns = exclude_patterns or ['tests/', 'test_', 'venv/', 'site-packages/']
        self.max_depth = max_depth

        self.traced_files: Dict[str, Set[int]] = defaultdict(set)
        self.function_calls: List[Tuple[str, str, int]] = []
        self.call_stack: List[Tuple[str, str, int]] = []

        self.is_tracing = False

    def should_trace_file(self, filename: str) -> bool:
        """Return True if the file should be traced."""
        if not filename:
            return False

        try:
            file_path = Path(filename).resolve()

            import sysconfig
            stdlib_paths = [sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')]
            for stdlib_path in stdlib_paths:
                if stdlib_path and str(file_path).startswith(stdlib_path):
                    return False

            if 'site-packages' in str(file_path) or 'dist-packages' in str(file_path):
                return False

            if not str(file_path).startswith(str(self.repo_path)):
                return False

            relative_path = str(file_path.relative_to(self.repo_path))
            for pattern in self.exclude_patterns:
                if pattern in relative_path:
                    return False

            if '__pycache__' in str(file_path) or file_path.suffix == '.pyc':
                return False

            return True
        except (ValueError, OSError):
            return False

    def trace_function(self, frame, event, arg):
        """Callback for sys.settrace."""
        if not self.is_tracing:
            return

        filename = frame.f_code.co_filename

        if not self.should_trace_file(filename):
            return self.trace_function

        if self.max_depth is not None and len(self.call_stack) >= self.max_depth:
            return self.trace_function

        lineno = frame.f_lineno
        func_name = frame.f_code.co_name

        if event == 'call':
            self.function_calls.append((filename, func_name, lineno))
            self.call_stack.append((filename, func_name, lineno))
            self.traced_files[filename].add(lineno)

        elif event == 'line':
            self.traced_files[filename].add(lineno)

        elif event == 'return':
            if self.call_stack and self.call_stack[-1][0] == filename:
                self.call_stack.pop()

        return self.trace_function

    def start_trace(self):
        """Start tracing."""
        self.is_tracing = True
        sys.settrace(self.trace_function)

    def stop_trace(self):
        """Stop tracing."""
        self.is_tracing = False
        sys.settrace(None)

    def get_trace_summary(self) -> Dict:
        """Return a trace summary dict."""
        file_coverage = {}
        for filepath, lines in self.traced_files.items():
            rel_path = str(Path(filepath).relative_to(self.repo_path))

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()

                executed_code = {}
                for line_num in sorted(lines):
                    if 0 < line_num <= len(file_lines):
                        executed_code[line_num] = file_lines[line_num - 1].rstrip()

                file_coverage[rel_path] = {
                    'absolute_path': filepath,
                    'executed_lines': sorted(lines),
                    'total_executed_lines': len(lines),
                    'executed_code': executed_code
                }
            except Exception as e:
                file_coverage[rel_path] = {
                    'absolute_path': filepath,
                    'executed_lines': sorted(lines),
                    'total_executed_lines': len(lines),
                    'error': str(e)
                }

        function_call_counts = defaultdict(int)
        for filepath, func_name, _ in self.function_calls:
            rel_path = str(Path(filepath).relative_to(self.repo_path))
            key = f"{rel_path}::{func_name}"
            function_call_counts[key] += 1

        return {
            'file_coverage': file_coverage,
            'function_calls': dict(function_call_counts),
            'total_files': len(self.traced_files),
            'total_function_calls': len(self.function_calls)
        }

    def save_trace(self, output_path: str):
        """Save trace results to a JSON file."""
        summary = self.get_trace_summary()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary


def trace_test_execution(repo_path: str, test_function, output_path: str = None) -> Dict:
    """
    Trace a single test function's execution.

    Args:
        repo_path: Repository root path.
        test_function: Test function to execute.
        output_path: Output path for trace results.

    Returns:
        Trace summary dict.
    """
    tracer = CodeTracer(repo_path)

    try:
        tracer.start_trace()
        test_function()
    except Exception as e:
        print(f"Test execution failed: {e}")
        raise
    finally:
        tracer.stop_trace()

    if output_path:
        return tracer.save_trace(output_path)
    else:
        return tracer.get_trace_summary()


if __name__ == "__main__":
    import unittest

    repo_path = "/path/to/your/repo"

    sys.path.insert(0, repo_path)
    from tests.test_parser import TestParser

    test_case = TestParser()

    def run_test():
        test_case.test_parse_empty()

    summary = trace_test_execution(
        repo_path=repo_path,
        test_function=run_test,
        output_path="./traces/test_parse_empty_trace.json"
    )

    print(f"Traced {summary['total_files']} files")
    print(f"Total function calls: {summary['total_function_calls']}")
    print("\nFiles covered:")
    for file_path in summary['file_coverage'].keys():
        print(f"  - {file_path}")
