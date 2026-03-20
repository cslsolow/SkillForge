"""Use an LLM to analyze traced code and identify critical segments."""
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class CodeAnalyzer:
    """Analyze code and identify critical segments using an LLM."""

    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        """
        Args:
            api_key: API key (if None, read from OPENAI_API_KEY env var).
            api_base: API base URL (if None, read from OPENAI_API_BASE env var).
            model: Model name (if None, read from OPENAI_MODEL env var, default gpt-4).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")

        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
            print(f"Using custom API base: {self.api_base}")

        self.client = OpenAI(**client_kwargs)

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call the LLM and return the response text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise

    def analyze_test_description(self, test_name: str, test_code: str) -> str:
        """
        Convert test code to natural language description IN ENGLISH.

        Args:
            test_name: Test name.
            test_code: Test code.

        Returns:
            Natural language description of the test IN ENGLISH.
        """
        system_prompt = "You are a code analysis expert who excels at understanding test code intent. You MUST respond in English."

        prompt = f"""
Analyze the following test case and describe its purpose and content in concise natural language IN ENGLISH:

Test name: {test_name}

Test code:
```python
{test_code}
```

Provide:
1. Test purpose (summarize in one sentence)
2. Specific test scenarios
3. Expected behavior

Output in JSON format IN ENGLISH:
{{
    "purpose": "...",
    "scenario": "...",
    "expected_behavior": "..."
}}

IMPORTANT: All descriptions MUST be in English.
"""

        response = self._call_llm(prompt, system_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "purpose": response,
                "scenario": "",
                "expected_behavior": ""
            }

    def identify_critical_code(
        self,
        trace_summary: Dict,
        test_description: Dict,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Identify the most critical code segments in the trace.

        Args:
            trace_summary: Trace summary from the tracer.
            test_description: Natural-language description of the test.
            top_k: Number of top critical segments to return.

        Returns:
            List of critical segment dicts, each containing:
            {
                'file': file path,
                'start_line': start line,
                'end_line': end line,
                'code': code content,
                'reason': why it is critical,
                'confidence': confidence score (0-1)
            }
        """
        file_coverage = trace_summary.get('file_coverage', {})

        code_snippets = []
        for file_path, coverage_info in file_coverage.items():
            executed_code = coverage_info.get('executed_code', {})
            if not executed_code:
                continue

            line_groups = self._group_consecutive_lines(executed_code.keys())

            for group in line_groups:
                code_lines = [executed_code[line] for line in group]
                snippet = {
                    'file': file_path,
                    'start_line': min(group),
                    'end_line': max(group),
                    'code': '\n'.join(code_lines),
                    'line_count': len(group)
                }
                code_snippets.append(snippet)

        system_prompt = """You are a code analysis expert. Given the test purpose, identify the most critical code segments executed during the test. Critical segments are typically:
1. Core logic that directly implements the tested functionality
2. Code containing important data processing or transformations
3. Code containing key conditional checks or control flow
4. Not utility functions, configuration code, or simple getters/setters
"""

        prompt = f"""Test description:
{json.dumps(test_description, indent=2, ensure_ascii=False)}

The following code segments were executed during the test. Identify the most critical {top_k} segments.

Code segment list:
"""

        for i, snippet in enumerate(code_snippets[:20]):
            prompt += f"""
---
Segment {i+1}:
File: {snippet['file']}
Lines: {snippet['start_line']}-{snippet['end_line']}
Code:
```python
{snippet['code']}
```
"""

        prompt += f"""
Analyze and return the {top_k} most critical segments. For each segment, explain:
1. Why it is critical
2. Its role in achieving the test purpose
3. Confidence score (0.0-1.0)

Return a JSON array where each element has this format:
[
    {{
        "snippet_id": <segment number (1-based)>,
        "file": "<file path>",
        "start_line": <start line number>,
        "end_line": <end line number>,
        "reason": "<why it is critical>",
        "role": "<role in the test>",
        "confidence": 0.9
    }}
]
"""

        response = self._call_llm(prompt, system_prompt)

        try:
            critical_segments = json.loads(response)

            results = []
            for segment in critical_segments[:top_k]:
                snippet_id = segment.get('snippet_id', 1) - 1
                if 0 <= snippet_id < len(code_snippets):
                    snippet = code_snippets[snippet_id]
                    results.append({
                        'file': snippet['file'],
                        'start_line': snippet['start_line'],
                        'end_line': snippet['end_line'],
                        'code': snippet['code'],
                        'reason': segment.get('reason', ''),
                        'role': segment.get('role', ''),
                        'confidence': segment.get('confidence', 0.5)
                    })

            return results

        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {response}")
            sorted_snippets = sorted(code_snippets, key=lambda x: x['line_count'], reverse=True)
            return [{
                'file': s['file'],
                'start_line': s['start_line'],
                'end_line': s['end_line'],
                'code': s['code'],
                'reason': 'Fallback: longest code segment',
                'role': 'Unknown',
                'confidence': 0.3
            } for s in sorted_snippets[:top_k]]

    def _group_consecutive_lines(self, line_numbers: List[int]) -> List[List[int]]:
        """Group consecutive line numbers into sublists."""
        if not line_numbers:
            return []

        sorted_lines = sorted(line_numbers)
        groups = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if line == current_group[-1] + 1:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]

        groups.append(current_group)
        return groups

    def save_analysis(self, analysis_result: Dict, output_path: str):
        """Save analysis results to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    analyzer = CodeAnalyzer(model="gpt-4")

    with open("./traces/test_parse_empty_trace.json", 'r') as f:
        trace_summary = json.load(f)

    test_code = """
    def test_parse_empty(self):
        with self.assertRaises(ParseError):
            parse_one("")
    """

    test_description = analyzer.analyze_test_description(
        "test_parse_empty",
        test_code
    )

    print("Test description:", json.dumps(test_description, indent=2, ensure_ascii=False))

    critical_code = analyzer.identify_critical_code(
        trace_summary,
        test_description,
        top_k=3
    )

    print(f"\nFound {len(critical_code)} critical code segments:")
    for i, segment in enumerate(critical_code):
        print(f"\n{i+1}. {segment['file']} (lines {segment['start_line']}-{segment['end_line']})")
        print(f"   Reason: {segment['reason']}")
        print(f"   Confidence: {segment['confidence']}")
