"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.utils.bm25_retriever import BM25Retriever
from minisweagent.utils.experience import (
    format_experience,
    is_synthesized_experience_payload,
    num_from_instance_id,
)
from minisweagent.utils.log import logger

SOURCE_EXTENSIONS = (".tsx", ".jsx", ".py", ".go", ".ts", ".js")
_SOURCE_EXT_PATTERN = "|".join(re.escape(ext) for ext in SOURCE_EXTENSIONS)
_SOURCE_PATH_PATTERN = rf"(?:(?:/|\./)?[\w.\-@]+/)*[\w.\-@]+(?:{_SOURCE_EXT_PATTERN})"


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    env_knowledge_top_k: int = 5


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, experience_data: dict | None = None, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self._experience_data = experience_data
        self._synthesized_experience_injected_api_paths: set[str] = set()
        self._selected_synthesized_experience_instance_ids: set[str] = set()
        self._synthesized_experience_keypoints: list[dict[str, Any]] = []
        self._synthesized_experience_env_knowledge: list[dict[str, Any]] = []

        if is_synthesized_experience_payload(experience_data):
            current_id = getattr(self, "instance_id", "")
            current_num = num_from_instance_id(current_id)

            raw_keypoints = experience_data.get("keypoints", []) or []
            if current_id:
                self._synthesized_experience_keypoints = [
                    kp for kp in raw_keypoints
                    if num_from_instance_id(kp.get("original_instance_id", kp.get("instance_id", ""))) <= current_num
                ]
            else:
                self._synthesized_experience_keypoints = raw_keypoints

            raw_env = experience_data.get("env_knowledge", []) or []
            if current_id:
                self._synthesized_experience_env_knowledge = [
                    item for item in raw_env
                    if num_from_instance_id(item.get("original_instance_id", item.get("source_instance_id", ""))) <= current_num
                ]
            else:
                self._synthesized_experience_env_knowledge = raw_env

            task = self._augment_task_with_synthesized_experience_env(task)

        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()

        messages = list(self.messages)
        if getattr(self, "_experience_data", None) and not is_synthesized_experience_payload(self._experience_data):
            experience_text = format_experience(self._experience_data)
            messages.append({"role": "user", "content": experience_text})
        # logger.info(f"LLM messages:\n{messages[-1]}")
        response = self.model.query(messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        parsed_action = self.parse_action(response)
        output = self.execute_action(parsed_action)
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        self._maybe_inject_synthesized_experience_keypoints(parsed_action.get("action", ""))
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            submission_content = "".join(lines[1:]).strip()
            # Guardrail: if the agent claims submission but provides an error message instead of a diff
            if "fatal:" in submission_content or "error:" in submission_content:
                raise NonTerminatingException(
                    f"Submission failed with git error:\n{submission_content}\n"
                    "Please fix the git environment (e.g., run 'rm -f .git/index.lock') and try submitting again."
                )

            # Guardrail: if the agent claims submission but provides no patch diff, keep going.
            # For SWE-bench runs, an empty submission usually means it ran only `echo COMPLETE_TASK...`
            # (or there are no staged changes), which is not actionable for evaluation.
            if lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and not submission_content:
                raise NonTerminatingException(
                    "Submission was empty. Run the exact submission command:\n"
                    "```bash\n"
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n"
                    "```\n"
                    "Make sure the diff is non-empty before submitting."
                )
            raise Submitted("".join(lines[1:]))

    def _augment_task_with_synthesized_experience_env(self, task: str) -> str:
        if not self._synthesized_experience_env_knowledge:
            return task

        selected_knowledge = self._synthesized_experience_env_knowledge
        top_k = getattr(self.config, "env_knowledge_top_k", 0)
        if top_k > 0 and len(self._synthesized_experience_env_knowledge) > top_k:
            try:
                docs = []
                for item in self._synthesized_experience_env_knowledge:
                    content = f"{item.get('api_path', '')} {item.get('purpose', '')} {item.get('playbook', '')}"
                    docs.append(content)

                retriever = BM25Retriever(docs)
                top_indices = retriever.get_top_k(task, top_k)
                selected_knowledge = [self._synthesized_experience_env_knowledge[i] for i in top_indices]
                logger.info(
                    f"Filtered global diagnostic skills from {len(self._synthesized_experience_env_knowledge)} to {len(selected_knowledge)} using BM25"
                )
            except Exception as e:
                logger.warning(f"Failed to use BM25 for global diagnostic skill filtering: {e}")

        for item in selected_knowledge:
            if sid := item.get("source_instance_id"):
                self._selected_synthesized_experience_instance_ids.add(sid)

        self._synthesized_experience_selected_env_knowledge = selected_knowledge
        lines: list[str] = [task, "", "=== Global Diagnostic Skills ==="]
        for item in selected_knowledge:
            api_path = item.get("api_path", "")
            purpose = item.get("purpose", "")
            if not api_path or not purpose:
                continue
            lines.append(f"- {api_path}: {purpose}")
            playbook = item.get("playbook", "")
            if isinstance(playbook, str) and playbook.strip():
                lines.append(f"  - playbook: {playbook.strip()}")
        return "\n".join(lines).strip()

    def _maybe_inject_synthesized_experience_keypoints(self, action: str) -> None:
        if not self._synthesized_experience_keypoints:
            return
        referenced_files = self._extract_source_filepaths_from_action(action)
        if not referenced_files:
            return
        triggered_items: dict[str, list[dict[str, Any]]] = {}
        for item in self._synthesized_experience_keypoints:
            api_path = item.get("api_path")
            instance_id = item.get("instance_id")
            if not isinstance(api_path, str) or not api_path:
                continue
            file_part = api_path.split("::", 1)[0]
            if (
                file_part in referenced_files
                and instance_id in self._selected_synthesized_experience_instance_ids
                and api_path not in self._synthesized_experience_injected_api_paths
            ):
                triggered_items.setdefault(api_path, []).append(item)
        if not triggered_items:
            return

        lines: list[str] = ["=== Local Intervention Skills ==="]
        for api_path, items in sorted(triggered_items.items(), key=lambda kv: kv[0]):
            lines.append(f"API: {api_path}")
            for it in items:
                exp = it.get("experience_or_reflection", "")
                issue = it.get("issue", "")
                if isinstance(exp, str) and exp.strip():
                    if isinstance(issue, str) and issue.strip():
                        lines.append(f"- [{issue.strip()}] {exp.strip()}")
                    else:
                        lines.append(f"- {exp.strip()}")
            lines.append("")
            self._synthesized_experience_injected_api_paths.add(api_path)
        self.add_message("user", "\n".join(lines).strip())

    @staticmethod
    def _extract_source_filepaths_from_action(action: str) -> set[str]:
        # Match common relative/absolute source file paths in shell commands.
        candidates = set(re.findall(_SOURCE_PATH_PATTERN, action))
        normalized: set[str] = set()
        for p in candidates:
            for prefix in ("/testbed/", "/testbed", "./"):
                if p.startswith(prefix):
                    p = p[len(prefix) :]
                    break
            p = p.lstrip("/")
            if p.endswith(SOURCE_EXTENSIONS):
                normalized.add(p)
        return normalized
