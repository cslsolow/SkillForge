#!/usr/bin/env python3
"""Rewrite the core implementations of a repository's main features guided by its test cases.

Given a passing test, the generator identifies the functions it exercises, masks their
implementations, and asks an LLM to rewrite them from context alone — producing an
alternative implementation that typically fails the test (a synthetic bug).

Key characteristics:
1. Original code is strictly masked — the LLM never sees it.
2. Only surrounding context (before/after) is provided.
3. The LLM rewrites the implementation from context and test purpose.
4. Exact string replacement is used to avoid position drift.
"""

import os
import sys
import json
import subprocess
import re
import ast
import inspect
import importlib
import shlex
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from tracer import CodeTracer
from code_analyzer import CodeAnalyzer


@dataclass
class BugAttempt:
    segment_idx: int
    file: str
    start_line: int
    end_line: int
    original_code: str
    generated_code: str
    is_valid_bug: bool = False
    error_type: str = ""
    error_message: str = ""


@dataclass 
class SuccessfulBug:
    segment: Dict
    buggy_code: str
    original_code: str
    error_message: str


class DetailedLogger:
    def __init__(self, output_dir: Path, log_name: str = "process.log"):
        self.output_dir = output_dir
        self.log_file = output_dir / log_name
        self.logs = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] [{level}] {message}")
        print(message)
    
    def section(self, title: str):
        self.log(f"\n{'=' * 70}")
        self.log(f"  {title}")
        self.log("=" * 70)
    
    def subsection(self, title: str):
        self.log(f"\n{'─' * 50}")
        self.log(f"  {title}")
        self.log("─" * 50)
    
    def save(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.logs))


class StrictMaskBugGenerator:
    """Rewrite repository implementations guided by test cases to produce synthetic bugs."""
    
    def __init__(
        self,
        repo_path: str,
        api_key: str = None,
        api_base: str = None,
        model: str = "gpt-5-mini",
        python_path: str = None
    ):
        self.repo_path = Path(repo_path).resolve()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        if python_path:
            self.python_path = python_path
        else:
            venv_python = self.repo_path / ".venv" / "bin" / "python"
            if venv_python.exists():
                self.python_path = str(venv_python)
            else:
                self.python_path = sys.executable

        client_kwargs: dict = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        self.client = OpenAI(**client_kwargs)
        self.analyzer = CodeAnalyzer(api_key=self.api_key, api_base=self.api_base, model=model)
        
        self._parse_repo_info()
    
    def _parse_repo_info(self):
        try:
            result = subprocess.run(
                "git remote get-url origin",
                shell=True, cwd=self.repo_path,
                capture_output=True, text=True
            )
            remote_url = result.stdout.strip()
            
            if 'github.com' in remote_url:
                parts = remote_url.rstrip('.git').split('/')
                self.repo_owner = parts[-2]
                self.repo_name = parts[-1]
            else:
                self.repo_owner = "unknown"
                self.repo_name = self.repo_path.name
            
            result = subprocess.run(
                "git rev-parse HEAD",
                shell=True, cwd=self.repo_path,
                capture_output=True, text=True
            )
            self.base_commit = result.stdout.strip()
            self.version = "0.0.0"
            
        except Exception:
            self.repo_owner = "unknown"
            self.repo_name = self.repo_path.name
            self.base_commit = "unknown"
            self.version = "0.0.0"
    
    def _call_llm(self, prompt: str, system_prompt: str = "", temperature: float = 0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _verify_test_passes(self, test_command: str, logger: DetailedLogger) -> bool:
        logger.log("Verifying test passes on original code...")

        # Replace only the python command at the start
        if test_command.startswith('python '):
            test_command_fixed = f'{self.python_path} ' + test_command[7:]
        else:
            test_command_fixed = test_command

        try:
            cmd_list = shlex.split(test_command_fixed)
        except ValueError as e:
            logger.log(f"  ✗ Failed to parse command: {e}", "ERROR")
            return False
        
        result = subprocess.run(
            cmd_list,
            shell=False,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.log("  ✓ Test PASSES on original code")
            return True
        else:
            stderr = result.stderr[:500]
            if 'ImportError' in stderr or 'ModuleNotFoundError' in stderr or 'cannot import' in stderr:
                logger.log(f"  ✗ Test FAILS on original code (Import Error - Environment Issue)!", "ERROR")
            else:
                logger.log(f"  ✗ Test FAILS on original code!", "ERROR")
            logger.log(f"  Command: {' '.join(cmd_list)}")
            logger.log(f"  stderr: {stderr}")
            return False
    
    def generate_bug(
        self,
        test_module: str,
        test_class: str,
        test_method: str,
        output_dir: str,
        top_k: int = 5,
        max_attempts_per_segment: int = 3,
        swebench_metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        
        test_name = test_method.replace("test_", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        test_output_dir = Path(output_dir) / f"{test_name}_{timestamp}"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = DetailedLogger(test_output_dir)
        
        if test_class:
            test_full_name = f"{test_module}::{test_class}::{test_method}"
            test_command = f"python -m pytest {test_module.replace('.', '/')}.py::{test_class}::{test_method} -x"
        else:
            test_full_name = f"{test_module}::::{test_method}"
            test_command = f"python -m pytest {test_module.replace('.', '/')}.py::{test_method} -x"
        
        logger.section(f"STRICT MASK BUG GENERATION")
        logger.log(f"Test: {test_full_name}")
        logger.log(f"Output: {test_output_dir}")
        logger.log(f"Mode: STRICT MASK (LLM cannot see original code)")
        
        if not self._verify_test_passes(test_command, logger):
            logger.save()
            return None
        
        logger.section("PHASE 1: TRACE TEST EXECUTION")
        trace_data = self._trace_test(test_module, test_class, test_method, logger)
        
        if not trace_data:
            logger.save()
            return None
        
        with open(test_output_dir / "01_trace.json", 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        logger.section("PHASE 2: EXTRACT CODE SEGMENTS")
        all_segments = self._extract_segments_with_context(trace_data, logger)
        
        with open(test_output_dir / "02_all_segments.json", 'w', encoding='utf-8') as f:
            json.dump(all_segments, f, indent=2, ensure_ascii=False)
        
        if not all_segments:
            logger.save()
            return None
        
        logger.section("PHASE 3: IDENTIFY CRITICAL SEGMENTS")
        critical_segments, llm_input, llm_output = self._identify_critical_segments(
            all_segments, trace_data.get('test_description', {}), top_k, logger
        )
        
        with open(test_output_dir / "03_llm_analysis.txt", 'w', encoding='utf-8') as f:
            f.write(f"=== INPUT ===\n{llm_input}\n\n=== OUTPUT ===\n{llm_output}")
        with open(test_output_dir / "03_critical_segments.json", 'w', encoding='utf-8') as f:
            json.dump(critical_segments, f, indent=2, ensure_ascii=False)
        
        if not critical_segments:
            logger.save()
            return None
        
        logger.section("PHASE 4: STRICT MASK CODE GENERATION")
        logger.log("LLM will NOT see the original code, only context!")

        test_description = trace_data.get('test_description', {})
        test_code = trace_data.get('test_code', '')
        other_tests = trace_data.get('other_tests', [])

        all_attempts: List[BugAttempt] = []
        successful_bugs: List[SuccessfulBug] = []
        
        for seg_idx, segment in enumerate(critical_segments):
            logger.subsection(f"Processing segment {seg_idx + 1}/{len(critical_segments)}")
            logger.log(f"File: {segment['file']}")
            logger.log(f"Lines: {segment['start_line']}-{segment['end_line']}")
            logger.log(f"Line count: {segment['line_count']}")
            logger.log(f"\n[MASKED - Original code hidden from LLM]")
            logger.log(f"Original code (for logging only):\n{segment['original_code']}")
            
            segment_has_bug = False
            
            for attempt_num in range(max_attempts_per_segment):
                if segment_has_bug:
                    break
                
                logger.log(f"\n  Attempt {attempt_num + 1}/{max_attempts_per_segment}...")
                
                generated_code, prompt_sent = self._generate_code_strict_mask(
                    segment, test_description, attempt_num, logger
                )
                
                attempt = BugAttempt(
                    segment_idx=seg_idx + 1,
                    file=segment['file'],
                    start_line=segment['start_line'],
                    end_line=segment['end_line'],
                    original_code=segment['original_code'],
                    generated_code=generated_code
                )
                
                if not generated_code:
                    logger.log("    Failed to generate code")
                    attempt.error_type = "generation_failed"
                    all_attempts.append(attempt)
                    continue
                
                if generated_code.strip() == segment['original_code'].strip():
                    logger.log("    Same as original (unlikely in strict mask mode)")
                    attempt.error_type = "same_as_original"
                    all_attempts.append(attempt)
                    continue
                
                logger.log(f"    Generated code:\n{generated_code}")

                result = self._verify_bug(
                    segment['absolute_path'],
                    segment['original_code'],
                    generated_code,
                    test_command,
                    logger,
                    segment['start_line'],
                    segment['end_line']
                )
                
                attempt.is_valid_bug = result['is_logic_bug']
                attempt.error_type = result['error_type']
                attempt.error_message = result['error_message'][:2000]
                all_attempts.append(attempt)
                
                if result['is_logic_bug']:
                    logger.log(f"    ✓✓ VALID LOGIC BUG! (test: PASS → FAIL)")
                    successful_bugs.append(SuccessfulBug(
                        segment=segment,
                        buggy_code=generated_code,
                        original_code=segment['original_code'],
                        error_message=result['error_message']
                    ))
                    segment_has_bug = True
                elif result['error_type'] == 'syntax_error':
                    logger.log(f"    ✗ Syntax error (rejected)")
                else:
                    logger.log(f"    ✗ Test still passes")
        
        with open(test_output_dir / "04_all_attempts.json", 'w', encoding='utf-8') as f:
            json.dump([asdict(a) for a in all_attempts], f, indent=2, ensure_ascii=False)
        
        logger.section("PHASE 5: COMBINE AND OUTPUT")
        
        if not successful_bugs:
            logger.log("No valid logic bugs found!", "ERROR")
            
            failure_analysis = {
                "total_attempts": len(all_attempts),
                "syntax_errors": len([a for a in all_attempts if a.error_type == 'syntax_error']),
                "test_passed": len([a for a in all_attempts if a.error_type == 'test_passed']),
                "other": len([a for a in all_attempts if a.error_type not in ['syntax_error', 'test_passed']])
            }
            with open(test_output_dir / "04_failure_analysis.json", 'w') as f:
                json.dump(failure_analysis, f, indent=2)
            
            logger.save()
            return None
        
        logger.log(f"Found {len(successful_bugs)} valid bugs to combine")

        result = self._generate_combined_output(
            successful_bugs, test_description, test_code, test_full_name,
            test_command, all_attempts, test_output_dir, logger, swebench_metadata,
            other_tests
        )
        
        root_output_dir = Path(output_dir)
        instances_file = root_output_dir / "instances.jsonl"
        with open(instances_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.log(f"Appended to: {instances_file}")
        
        logger.section("COMPLETE")
        logger.log(f"Instance ID: {result['instance_id']}")
        logger.log(f"Bugs combined: {len(successful_bugs)}")
        logger.save()
        
        return result
    
    def _trace_test(self, test_module: str, test_class: str, test_method: str, logger: DetailedLogger) -> Optional[Dict]:
        logger.log(f"Tracing: {test_module}::{test_class}::{test_method}")
        
        try:
            if test_class:
                import_test_class = f"test_class_obj = getattr(module, '{test_class}')\ntest_instance = test_class_obj()\nmethod = getattr(test_instance, '{test_method}')"
            else:
                import_test_class = f"method = getattr(module, '{test_method}')"
            
            get_source_script = f"""
import sys
import inspect
sys.path.insert(0, '{self.repo_path}')

# Import module
module = __import__('{test_module}', fromlist=[''])

# Get test method
{import_test_class}

# Print source code
print(inspect.getsource(method))
"""
            
            result = subprocess.run(
                [self.python_path, '-c', get_source_script],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.log(f"Error getting test source: {result.stderr}", "ERROR")
                return None
            
            test_code = result.stdout
            logger.log(f"\nTest code:\n{test_code}")
            
            test_description = self.analyzer.analyze_test_description(test_method, test_code)

            # Use pytest to collect all tests in the same file
            test_file = test_module.replace('.', '/') + '.py'
            test_file_path = self.repo_path / test_file

            if test_file_path.exists():
                collect_cmd = [self.python_path, '-m', 'pytest', '--collect-only', '-q', str(test_file_path)]
                try:
                    collect_result = subprocess.run(
                        collect_cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=30
                    )
                    # Parse collected test names from pytest output
                    all_tests = []
                    for line in collect_result.stdout.splitlines():
                        if '::' in line and not line.startswith(' '):
                            all_tests.append(line.strip())

                    # Filter out the current test
                    other_tests = [t for t in all_tests if test_method not in t]
                    logger.log(f"Found {len(other_tests)} other tests in {test_file}")
                except Exception as e:
                    logger.log(f"Failed to collect tests: {e}", "WARN")
                    other_tests = []
            else:
                other_tests = []

            # Build test path for tracing
            if test_class:
                test_path = f"{test_file}::{test_class}::{test_method}"
            else:
                test_path = f"{test_file}::{test_method}"

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                trace_output = f.name

            env = os.environ.copy()
            env['PYTHONPATH'] = f"{self.repo_path}:{env.get('PYTHONPATH', '')}"

            # Run the target test with coverage tracing using pytest's --trace option
            # If pytest_tracer_plugin is not available, use coverage.py as fallback
            trace_cmd = [
                self.python_path, '-m', 'coverage', 'run',
                '--source', str(self.repo_path),
                '-m', 'pytest', test_path, '-xvs'
            ]

            result = subprocess.run(
                trace_cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=1000,
                env=env
            )

            # Generate coverage report in JSON format
            coverage_json = self.repo_path / '.coverage_report.json'
            subprocess.run(
                [self.python_path, '-m', 'coverage', 'json', '-o', str(coverage_json)],
                cwd=self.repo_path,
                capture_output=True,
                timeout=60
            )

            trace_summary = {'total_files': 0, 'file_coverage': {}}
            if coverage_json.exists():
                try:
                    coverage_data = json.loads(coverage_json.read_text())
                    files = coverage_data.get('files', {})

                    for filepath, file_data in files.items():
                        rel_path = str(Path(filepath).relative_to(self.repo_path))
                        executed_lines = file_data.get('executed_lines', [])

                        if executed_lines:
                            executed_code = {}
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    file_lines = f.readlines()
                                for line_num in executed_lines:
                                    if 0 < line_num <= len(file_lines):
                                        executed_code[str(line_num)] = file_lines[line_num - 1].rstrip()
                            except Exception:
                                pass

                            trace_summary['file_coverage'][rel_path] = {
                                'absolute_path': filepath,
                                'executed_lines': executed_lines,
                                'total_executed_lines': len(executed_lines),
                                'executed_code': executed_code
                            }

                    trace_summary['total_files'] = len(trace_summary['file_coverage'])
                    coverage_json.unlink()
                except Exception as e:
                    logger.log(f"Failed to parse coverage data: {e}", "WARN")

            if os.path.exists(trace_output):
                os.unlink(trace_output)
            
            logger.log(f"Traced {trace_summary.get('total_files', 0)} files")

            return {
                'test_module': test_module,
                'test_class': test_class,
                'test_method': test_method,
                'test_code': test_code,
                'test_description': test_description,
                'trace_summary': trace_summary,
                'other_tests': other_tests
            }

        except Exception as e:
            logger.log(f"Error: {e}", "ERROR")
            import traceback
            logger.log(traceback.format_exc(), "ERROR")
            return None
    
    def _extract_segments_with_context(
        self,
        trace_data: Dict,
        logger: DetailedLogger,
        min_lines: int = 3,
        max_lines: int = 30,
        context_lines: int = 10
    ) -> List[Dict]:
        """Extract segments directly from trace data without heuristic or hybrid strategies."""
        
        file_coverage = trace_data.get('trace_summary', {}).get('file_coverage', {})
        all_segments = []
        
        for rel_path, coverage_info in file_coverage.items():
            abs_path = coverage_info.get('absolute_path', '')
            executed_code = coverage_info.get('executed_code', {})
            
            if not executed_code or not abs_path:
                continue
            
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    file_lines = file_content.splitlines(keepends=True)
            except:
                continue
            
            executed_lines = sorted(int(l) for l in executed_code.keys())
            if not executed_lines:
                continue

            start_line = min(executed_lines)
            end_line = max(executed_lines)

            segment = self._create_segment_from_lines(
                start_line, end_line, rel_path, abs_path,
                file_lines, executed_code, context_lines
            )
            if segment:
                segment['extraction_type'] = 'trace_span'
                all_segments.append(segment)
                logger.log(
                    f"  Extracted trace span: {rel_path} lines {start_line}-{end_line} "
                    f"({segment['line_count']} lines)"
                )

        logger.log(f"Extracted {len(all_segments)} segments directly from trace spans")
        
        return all_segments
    
    def _create_complete_function_segment(
        self,
        func_info: Dict,
        rel_path: str,
        abs_path: str,
        file_lines: List[str],
        context_lines: int
    ) -> Optional[Dict]:
        """Create a segment from a complete function definition."""
        
        start_line = func_info['start_line']
        end_line = func_info['end_line']
        
        original_code = ''.join(file_lines[start_line-1:end_line])
        
        context_start = max(0, start_line - 1 - context_lines)
        context_end = min(len(file_lines), end_line + context_lines)
        
        context_before = ''.join(file_lines[context_start:start_line-1])
        context_after = ''.join(file_lines[end_line:context_end])
        
        first_line = file_lines[start_line-1] if start_line-1 < len(file_lines) else ""
        base_indent = len(first_line) - len(first_line.lstrip())

        return {
            'file': rel_path,
            'absolute_path': abs_path,
            'start_line': start_line,
            'end_line': end_line,
            'line_count': end_line - start_line + 1,
            'original_code': original_code.rstrip('\n'),
            'context_before': context_before,
            'context_after': context_after,
            'base_indent': base_indent,
            'function_name': func_info['name'],
            'function_type': func_info['type'],
            'has_docstring': func_info['has_docstring']
        }
    
    def _create_segment_from_lines(
        self,
        start_line: int,
        end_line: int,
        rel_path: str,
        abs_path: str,
        file_lines: List[str],
        executed_code: Dict,
        context_lines: int
    ) -> Optional[Dict]:
        """Create a segment from a line-number range."""
        
        original_code = ''.join(file_lines[start_line-1:end_line])
        
        context_start = max(0, start_line - 1 - context_lines)
        context_end = min(len(file_lines), end_line + context_lines)
        
        context_before = ''.join(file_lines[context_start:start_line-1])
        context_after = ''.join(file_lines[end_line:context_end])
        
        first_line = file_lines[start_line-1] if start_line-1 < len(file_lines) else ""
        base_indent = len(first_line) - len(first_line.lstrip())

        return {
            'file': rel_path,
            'absolute_path': abs_path,
            'start_line': start_line,
            'end_line': end_line,
            'line_count': end_line - start_line + 1,
            'original_code': original_code.rstrip('\n'),
            'context_before': context_before,
            'context_after': context_after,
            'base_indent': base_indent,
            'code_summary': executed_code
        }
    
    def _identify_critical_segments(
        self,
        segments: List[Dict],
        test_description: Dict,
        top_k: int,
        logger: DetailedLogger
    ) -> Tuple[List[Dict], str, str]:
        """Identify top-k critical code segments by sending all traced segments to the LLM."""
        
        system_prompt = """You are a code analysis expert. Identify the core logic segments of a test case.
Select those that:
1. Directly affect the test outcome
2. Contain conditional checks, exception raising, or return values
3. Are not simple initialization or utility functions"""

        segments_text = ""
        for i, seg in enumerate(segments):
            segments_text += f"""
---
Segment {i+1}:
File: {seg['file']}
Lines: {seg['start_line']}-{seg['end_line']} ({seg['line_count']} lines)
```python
{seg['original_code']}
```
"""
        
        prompt = f"""Test purpose: {test_description.get('purpose', 'N/A')}
Test scenario: {test_description.get('scenario', 'N/A')}

Segment list:
{segments_text}

Select the {top_k} most critical segments — those whose modification is most likely to cause the test to fail.

Evaluation criteria:
1. Core logic directly involved in the test scenario
2. Contains key conditional checks, data processing, or return values
3. Not a simple helper function or configuration code

For each selected segment, assess its importance (confidence):
- 0.9-1.0: Core logic, almost certain to affect the test
- 0.7-0.89: Important logic, likely to affect the test
- 0.5-0.69: Related logic, may affect the test

Return a JSON array sorted by importance:
```json
[
    {{"segment_id": <number between 1 and {len(segments)}>, "reason": "<detailed explanation>", "confidence": <value between 0.5-1.0>}}
]
```"""
        
        response = self._call_llm(prompt, system_prompt)
        rankings = self._parse_rankings(response)
        
        if not rankings:
            logger.log(f"  WARNING: No rankings parsed from response", "WARN")
            logger.log(f"  Response preview: {response[:300]}...", "WARN")
        else:
            logger.log(f"  Parsed {len(rankings)} rankings")
        
        critical_segments = []
        for rank in rankings[:top_k]:
            seg_id = rank.get('segment_id', 0) - 1
            if 0 <= seg_id < len(segments):
                seg = segments[seg_id].copy()
                seg['reason'] = rank.get('reason', '')
                seg['confidence'] = rank.get('confidence', 0)
                critical_segments.append(seg)
                logger.log(f"  Selected: segment_id={rank.get('segment_id')}, confidence={rank.get('confidence'):.2f}")
                logger.log(f"    -> {seg['file']} lines {seg['start_line']}-{seg['end_line']}")
            else:
                logger.log(f"  WARNING: Invalid segment_id {rank.get('segment_id')} (valid range: 1-{len(segments)})", "WARN")
        
        return critical_segments, prompt, response
    
    def _parse_rankings(self, response: str) -> List[Dict]:
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                result = json.loads(matches[0])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to parse JSON from code block: {e}")
        
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                if isinstance(result, list):
                    return result
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse JSON array: {e}")
            print(f"DEBUG: Attempted to parse: {response[start:end][:200]}...")
        
        print(f"DEBUG: No valid rankings found in response")
        print(f"DEBUG: Response preview: {response[:500]}...")
        return []
    
    def _generate_code_strict_mask(
        self,
        segment: Dict,
        test_description: Dict,
        attempt: int,
        logger: DetailedLogger
    ) -> Tuple[str, str]:
        """
        Generate replacement code in strict-mask mode:
        - The original code is never shown.
        - Only surrounding context is provided.
        - Line count and indentation requirements are communicated.
        - Strategy is adjusted based on extraction type.
        """
        
        line_count = segment['line_count']
        base_indent = segment['base_indent']
        indent_str = ' ' * base_indent
        
        context_before = segment['context_before']
        context_after = segment['context_after']
        
        extraction_type = segment.get('extraction_type', 'executed_fragment')
        is_complete_function = extraction_type == 'complete_function'

        if is_complete_function:
            func_name = segment.get('function_name', 'unknown')
            has_docstring = segment.get('has_docstring', False)
            extra_context = f"Function name: {func_name}"
            if has_docstring:
                extra_context += " (has docstring — use context to infer purpose)"
        else:
            extra_context = ""

        if is_complete_function:
            system_prompt = f"""You are a Python developer. You need to implement a complete function/method.

Key rules:
1. You cannot see the original implementation — infer its purpose from context (function signature, docstring, surrounding code).
2. Base indentation is {base_indent} spaces (the indentation of the function definition line).
3. Code must be syntactically correct, runnable, and semantically complete.
4. {extra_context}

For this complete function/method, you must:
- Understand the function's intent (infer from name, parameters, and context).
- Implement the core functional logic.
- The generated code must be a semantically complete block — do not use pass as a placeholder.

Indentation example:
{indent_str}def function_name(...):
{indent_str}    # function body (+4 spaces)
{indent_str}    return result"""
        else:
            system_prompt = f"""You are a Python developer. You need to fill in the code marked as [MASKED] based on context.

Key rules:
1. You cannot see the original code — infer it from context only.
2. Base indentation is {base_indent} spaces.
3. Code must be syntactically correct, runnable, and semantically complete.
4. The generated code must be a natural code block — do not fill with multiple pass statements.

Indentation example:
{indent_str}first line of code
{indent_str}    deeper indentation (+4 spaces) if needed
{indent_str}second line of code"""

        if is_complete_function:
            func_signature = ""
            original_lines = segment['original_code'].split('\n')
            for line in original_lines[:3]:
                if 'def ' in line or 'class ' in line or 'async def ' in line:
                    func_signature = line.strip()
                    break

            prompt = f"""
Implement the complete function marked as [MASKED] below.

File: {segment['file']}
Function: {segment.get('function_name', 'unknown')}
Location: lines {segment['start_line']} to {segment['end_line']} (original ~{line_count} lines; you may adjust as needed)
Base indentation: {base_indent} spaces
{f'Function signature hint: {func_signature}' if func_signature else ''}

Context (before the function definition):
```python
{context_before}```

[MASKED: implement the complete function here]

Context (after the function definition):
```python
{context_after}```

Test purpose: {test_description.get('purpose', 'unknown')}
Test scenario: {test_description.get('scenario', 'unknown')}

Implement the function based on context:
- Generate a semantically complete block including the function signature and body.
- Maintain base indentation of {base_indent} spaces.
- Do not fill with pass or comments — implement actual logic.
- Output only the code block, no explanations.
"""
        else:
            context_hints = []
            if 'if ' in context_before or 'for ' in context_before or 'while ' in context_before:
                context_hints.append('Note: the preceding code contains control-flow statements')
            if 'return' in context_after:
                context_hints.append('Note: the following code contains a return statement')
            if 'raise' in context_after or 'except' in context_after:
                context_hints.append('Note: the following code involves exception handling')

            hints_text = '\n'.join(f'- {h}' for h in context_hints) if context_hints else ''

            prompt = f"""
Fill in the [MASKED] section in the code below.

File: {segment['file']}
Location: lines {segment['start_line']} to {segment['end_line']} (original ~{line_count} lines; adjust as semantics require)
Base indentation: {base_indent} spaces

Context (code before [MASKED]):
```python
{context_before}```

[MASKED: fill in this code segment]

Context (code after [MASKED]):
```python
{context_after}```

Test purpose: {test_description.get('purpose', 'unknown')}
Test scenario: {test_description.get('scenario', 'unknown')}
"""

            if hints_text:
                prompt += f"\nContext hints:\n{hints_text}\n"

            prompt += """
Generate a semantically complete code block:
- Maintain base indentation of {base_indent} spaces.
- The code must flow naturally with the surrounding context.
- Do not fill with multiple pass statements — implement actual logic.
- Output only the code block, no explanations.
"""
        
        logger.log(f"    [STRICT MASK] Type: {extraction_type}")
        logger.log(f"    Context before: {len(context_before)} chars")
        logger.log(f"    Context after: {len(context_after)} chars")
        logger.log(f"    Original: {line_count} lines, base_indent={base_indent}")
        logger.log(f"    Note: Line count is reference only, semantic completeness is priority")
        
        response = self._call_llm(prompt, system_prompt, temperature= 0 + attempt * 0.1)
        code = self._extract_code(response)
        
        original_code = segment['original_code']
        original_first_line = original_code.strip().split('\n')[0].strip()
        generated_first_line = code.strip().split('\n')[0].strip() if code.strip() else ''
        
        if (original_first_line.startswith(('def ', 'class ', 'async def ')) and 
            not generated_first_line.startswith(('def ', 'class ', 'async def '))):
            effective_base_indent = base_indent + 4
            logger.log(f"    [INFO] Original has def/class, generated doesn't. Adjusting indent to {effective_base_indent}")
        else:
            effective_base_indent = base_indent
        
        code = self._fix_indentation_semantic(code, effective_base_indent)
        
        return code, prompt
    
    def _fix_indentation(self, code: str, target_lines: int, base_indent: int) -> str:
        """Kept for compatibility; prefer _fix_indentation_semantic."""
        return self._fix_indentation_semantic(code, base_indent)
    
    def _fix_indentation_semantic(self, code: str, base_indent: int) -> str:
        """Fix code indentation while preserving original structure.

        Strategy:
        1. Detect the minimum indentation in the generated code
        2. Adjust all lines by the difference to match base_indent
        3. Preserve relative indentation between lines
        """

        if not code.strip():
            return ' ' * base_indent + 'pass'

        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return ' ' * base_indent + 'pass'

        # Find minimum indentation (excluding empty lines)
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

        # Calculate adjustment needed
        indent_adjustment = base_indent - min_indent

        # Apply adjustment to all lines, preserving relative indentation
        fixed_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                new_indent = max(0, current_indent + indent_adjustment)
                fixed_lines.append(' ' * new_indent + line.lstrip())
            else:  # Empty line
                fixed_lines.append('')

        return '\n'.join(fixed_lines)
    
    def _extract_code(self, response: str) -> str:
        code_block_pattern = r'```(?:python)?\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            code = matches[0].strip()
        else:
            code = response.strip()
        
        code = self._fix_incomplete_code(code)
        return code
    
    def _fix_incomplete_code(self, code: str) -> str:
        """Fix incomplete code such as unclosed brackets."""

        open_parens = code.count('(') - code.count(')')
        open_brackets = code.count('[') - code.count(']')
        open_braces = code.count('{') - code.count('}')
        
        if open_parens > 0 or open_brackets > 0 or open_braces > 0:
            if ' if ' in code and ' else ' not in code:
                code = code.rstrip() + ' else None'

            code = code + ')' * open_parens + ']' * open_brackets + '}' * open_braces
        
        return code

    def _verify_bug(
        self,
        file_path: str,
        original_code: str,
        generated_code: str,
        test_command: str,
        logger: DetailedLogger,
        start_line: int,
        end_line: int
    ) -> Dict:
        """Verify whether the generated code introduces a logic bug."""

        result = {
            'is_logic_bug': False,
            'error_type': '',
            'error_message': ''
        }

        with open(file_path, 'r', encoding='utf-8') as f:
            file_lines = f.readlines()

        original_file_content = ''.join(file_lines)

        try:
            # Use line-based replacement instead of string replacement
            if start_line < 1 or end_line > len(file_lines):
                result['error_type'] = 'invalid_line_range'
                result['error_message'] = f'Line range {start_line}-{end_line} out of bounds'
                return result

            # Replace lines based on line numbers
            modified_lines = (
                file_lines[:start_line-1] +
                [generated_code + '\n'] +
                file_lines[end_line:]
            )
            modified_content = ''.join(modified_lines)
            
            try:
                ast.parse(modified_content)
            except SyntaxError as e:
                result['error_type'] = 'syntax_error'
                result['error_message'] = str(e)
                return result
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            # Replace only the python command at the start
            if test_command.startswith('python '):
                test_command_fixed = f'{self.python_path} ' + test_command[7:]
            else:
                test_command_fixed = test_command
            test_result = subprocess.run(
                test_command_fixed,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if test_result.returncode != 0:
                result['is_logic_bug'] = True
                result['error_type'] = 'logic_bug'
                result['error_message'] = test_result.stdout + test_result.stderr
            else:
                result['error_type'] = 'test_passed'
                result['error_message'] = 'Test still passes'
                
        except subprocess.TimeoutExpired:
            result['error_type'] = 'timeout'
            result['is_logic_bug'] = False  # Timeout is not a reliable logic bug indicator
            result['error_message'] = 'Test timeout - may be environment issue or infinite loop'
        except Exception as e:
            result['error_type'] = 'error'
            result['error_message'] = str(e)
        finally:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(original_file_content)
        
        return result
    
    def _generate_combined_output(
        self,
        successful_bugs: List[SuccessfulBug],
        test_description: Dict,
        test_code: str,
        test_full_name: str,
        test_command: str,
        all_attempts: List[BugAttempt],
        output_dir: Path,
        logger: DetailedLogger,
        swebench_metadata: Optional[Dict] = None,
        other_tests: List[str] = None
    ) -> Dict:
        """Generate combined output from all successful bugs."""
        
        files_content: Dict[str, str] = {}
        for bug in successful_bugs:
            file_path = bug.segment['absolute_path']
            if file_path not in files_content:
                with open(file_path, 'r', encoding='utf-8') as f:
                    files_content[file_path] = f.read()

        buggy_contents: Dict[str, str] = {}
        for file_path, content in files_content.items():
            file_lines = content.splitlines(keepends=True)

            # Sort bugs by line number (descending) to avoid line number shifts
            bugs_for_file = [b for b in successful_bugs if b.segment['absolute_path'] == file_path]
            bugs_for_file.sort(key=lambda b: b.segment['start_line'], reverse=True)

            for bug in bugs_for_file:
                start_line = bug.segment['start_line']
                end_line = bug.segment['end_line']

                if start_line >= 1 and end_line <= len(file_lines):
                    file_lines = (
                        file_lines[:start_line-1] +
                        [bug.buggy_code + '\n'] +
                        file_lines[end_line:]
                    )

            buggy_contents[file_path] = ''.join(file_lines)
        
        logger.log("Verifying combined patch...")
        combined_works = False
        # Replace only the python command at the start
        if test_command.startswith('python '):
            test_command_fixed = f'{self.python_path} ' + test_command[7:]
        else:
            test_command_fixed = test_command
        try:
            for file_path, content in buggy_contents.items():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            result = subprocess.run(test_command_fixed, shell=True, cwd=self.repo_path,
                                   capture_output=True, text=True, timeout=60)
            combined_works = result.returncode != 0
            logger.log(f"  Combined patch: {'✓ breaks test' if combined_works else '✗ test passes'}")
        finally:
            for file_path, content in files_content.items():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        combined_buggy_patch = ""
        combined_gold_patch = ""
        
        for file_path in files_content:
            rel_path = [b.segment['file'] for b in successful_bugs if b.segment['absolute_path'] == file_path][0]
            
            buggy_diff = self._generate_diff(files_content[file_path], buggy_contents[file_path], rel_path)
            gold_diff = self._generate_diff(buggy_contents[file_path], files_content[file_path], rel_path)
            
            combined_buggy_patch += buggy_diff
            combined_gold_patch += gold_diff
        
        with open(output_dir / "05_combined_buggy.patch", 'w') as f:
            f.write(combined_buggy_patch)
        with open(output_dir / "05_combined_gold.patch", 'w') as f:
            f.write(combined_gold_patch)
        logger.log(f"Patches saved")
        
        test_error_output = ""
        try:
            for file_path, content in buggy_contents.items():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            result = subprocess.run(test_command_fixed, shell=True, cwd=self.repo_path,
                                   capture_output=True, text=True, timeout=60)
            test_error_output = result.stdout + result.stderr
        finally:
            for file_path, content in files_content.items():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        problem_statement = self._generate_issue(
            bugs=successful_bugs,
            test_description=test_description,
            test_code=test_code,
            buggy_patch=combined_buggy_patch,
            test_error_output=test_error_output,
            logger=logger
        )
        with open(output_dir / "05_issue.md", 'w', encoding='utf-8') as f:
            f.write(problem_statement)
        
        summary = {
            "mode": "strict_mask",
            "bugs_count": len(successful_bugs),
            "files_modified": list(set(b.segment['file'] for b in successful_bugs)),
            "total_attempts": len(all_attempts),
            "combined_patch_works": combined_works,
            "bugs": [
                {
                    "file": b.segment['file'],
                    "lines": f"{b.segment['start_line']}-{b.segment['end_line']}",
                    "reason": b.segment.get('reason', ''),
                    "original": b.original_code,
                    "buggy": b.buggy_code
                }
                for b in successful_bugs
            ]
        }
        with open(output_dir / "05_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        prefix = f"{self.repo_owner}__{self.repo_name}"
        if swebench_metadata and "instance_id" in swebench_metadata:
            orig_id = swebench_metadata["instance_id"]
            if "__" in orig_id:
                orig_id = orig_id.split("__")[-1]
            # Format: sympy__sympy-12419
            instance_id = f"{prefix}-{orig_id}"
        else:
            # Fallback: generate simple numeric ID
            instance_id = f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Use other tests in the same file as PASS_TO_PASS
        pass_to_pass = other_tests if other_tests else []

        instance = {
            "instance_id": instance_id,
            "repo": f"{self.repo_owner}/{self.repo_name}",
            "base_commit": self.base_commit,
            "patch": combined_gold_patch,
            "problem_statement": problem_statement,
            "hints_text": "",
            "created_at": datetime.now().isoformat(),
            "FAIL_TO_PASS": json.dumps([test_full_name]),
            "PASS_TO_PASS": json.dumps(pass_to_pass),
            "buggy_patch": combined_buggy_patch,
            "bug_count": len(successful_bugs),
            "modified_files": list(set(b.segment['file'] for b in successful_bugs))
        }
        
        if swebench_metadata:
            instance["swebench_source"] = {
                "original_instance_id": swebench_metadata.get("instance_id"),
                "original_base_commit": swebench_metadata.get("base_commit"),
                "note": "This bug was generated from a test case in SWE-bench Verified"
            }
        
        with open(output_dir / "06_instance.json", 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
        
        return instance
    
    def _generate_diff(self, old: str, new: str, file_path: str) -> str:
        import difflib
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, f"a/{file_path}", f"b/{file_path}")
        return ''.join(diff)
    
    def _generate_issue(
        self,
        bugs: List[SuccessfulBug],
        test_description: Dict,
        test_code: str,
        buggy_patch: str,
        test_error_output: str,
        logger: DetailedLogger
    ) -> str:
        """Generate a GitHub issue from the buggy patch and test failure output.

        Approach:
        1. Reverse-engineer the problems users would encounter from each bug's code changes.
        2. Integrate all bugs' impacts into a comprehensive problem description.
        3. Do not provide reproduction steps to avoid misleading the model.
        """
        
        # Build detailed analysis for each bug (for LLM to understand the root issues)
        bug_analyses = []
        for i, bug in enumerate(bugs, 1):
            # Analyze what problems this bug will cause
            bug_analyses.append(f"""
### Bug {i}: {bug.segment['file']}

**Location**: Lines {bug.segment['start_line']}-{bug.segment['end_line']}

**Correct Code** (Expected user behavior):
```python
{bug.original_code}
```

**Buggy Code** (Current version with bug):
```python
{bug.buggy_code}
```

**Potential Problems from This Change**: Please analyze what functional anomalies this code change will cause
""")
        
        bug_analyses_text = "\n".join(bug_analyses)
        
        error_lines = test_error_output.strip().split('\n')
        if len(error_lines) > 30:
            error_summary = '\n'.join(error_lines[-30:])
        else:
            error_summary = test_error_output.strip()
        
        prompt = f"""You are a technical documentation expert. Write a GitHub issue based on code changes and error information.

## Task Description

Based on the following information, reverse-engineer what problems users will encounter when using this library, and write an issue.

Analyze each bug's code changes, understand what problems they will cause individually, and then provide a comprehensive description.

## Feature Context

Test case description: {test_description.get('purpose', 'N/A')}

Related test code:
```python
{test_code}
```

## Bug Details ({len(bugs)} bugs total, all must be considered)

{bug_analyses_text}

## Actual Error Output When Running

```
{error_summary}
```

## Writing Requirements

1. Analyze each bug's impact - Infer what problems each bug will cause based on code changes
2. Comprehensive problem description - Integrate all bugs' impacts into a problem description from user's perspective
3. Write from user's viewpoint - Users don't know where the specific bugs are, can only describe observed abnormal behavior
4. Don't provide reproduction steps - Avoid giving potentially incorrect code examples
5. Expected vs Actual - Clearly contrast expected behavior and actual behavior
6. Can mention multiple issues - If different bugs cause different anomalies, mention them all

## Output Format

Use Markdown format, including:
- Title (starting with #, summarizing main problem)
- Problem description (describing encountered abnormalities, can include multiple aspects)
- Expected behavior
- Actual behavior (can list multiple abnormal phenomena)
- Error messages (if there are clear error types)

Do NOT include:
- Reproduction steps
- Specific code examples
- Environment information

Output the issue content directly without additional explanations.
"""
        
        system_prompt = """You are an experienced developer reporting a bug.
Your task is to reverse-engineer what problems users will encounter based on code changes.
Consider all bugs and comprehensively describe all possible abnormal phenomena.
Don't expose specific bug locations, only describe behavior users can observe."""
        
        issue_content = self._call_llm(prompt, system_prompt)
        
        if issue_content.startswith('```markdown'):
            issue_content = issue_content[len('```markdown'):].strip()
        if issue_content.startswith('```'):
            issue_content = issue_content[3:].strip()
        if issue_content.endswith('```'):
            issue_content = issue_content[:-3].strip()
        
        return issue_content


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Strict Mask Bug Generator")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--test-module", required=True)
    parser.add_argument("--test-class", required=True)
    parser.add_argument("--test-method", required=True)
    parser.add_argument("--output", default="./mask_output")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    
    args = parser.parse_args()
    
    generator = StrictMaskBugGenerator(repo_path=args.repo)
    
    result = generator.generate_bug(
        test_module=args.test_module,
        test_class=args.test_class,
        test_method=args.test_method,
        output_dir=args.output,
        top_k=args.top_k,
        max_attempts_per_segment=args.max_attempts
    )
    
    if result:
        print(f"\n✓ Success!")
        print(f"  Instance: {result['instance_id']}")
        print(f"  Bugs: {result.get('bug_count', 1)}")
        print(f"  Files: {result.get('modified_files', [])}")
    else:
        print("\n✗ Failed to generate valid logic bugs")
        sys.exit(1)


if __name__ == "__main__":
    main()
