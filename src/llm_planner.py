from __future__ import annotations

import json
import os
import re
from pathlib import Path
import urllib.error
import urllib.request

from planner import RuleBasedPlanner


def extract_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response."""
    cleaned = _strip_markdown_fence(text.strip())
    candidate = _first_json_object(cleaned)
    if candidate is None:
        raise ValueError("Could not find a JSON object in the LLM response.")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not parse LLM JSON output: {error}") from error


class LLMPlanner:
    """Optional LLM-backed planner with deterministic mock fallbacks."""

    def __init__(self, task: dict, capabilities: dict, mode: str = "auto"):
        _load_dotenv_if_present()
        self.task = task
        self.capabilities = capabilities
        self.mode = mode
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.mode_used = "unknown"
        self.last_error: str | None = None

    def generate_plan(self) -> dict:
        if self.mode in {"mock", "mock_bad"}:
            self.mode_used = "mock_bad"
            return self._mock_bad_plan()

        if self.mode == "mock_good":
            self.mode_used = "mock_good"
            return self._mock_good_plan()

        if self.mode in {"auto", "openai"}:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self.mode_used = "mock_bad"
                self.last_error = "OPENAI_API_KEY is not set; using mock_bad."
                return self._mock_bad_plan()

            try:
                self.mode_used = "openai"
                return self._openai_plan(api_key)
            except Exception as error:
                self.mode_used = "mock_bad_fallback"
                self.last_error = f"OpenAI-compatible request failed; using mock_bad. Error: {_safe_error_message(error)}"
                return self._mock_bad_plan()

        raise ValueError(f"Unsupported LLMPlanner mode '{self.mode}'.")

    def build_prompt(self) -> str:
        task_payload = {
            "task_id": self.task.get("task_id"),
            "instruction": self.task.get("instruction"),
            "initial_state": self.task.get("initial_state", []),
            "goal_state": self.task.get("goal_state", []),
            "objects": self.task.get("objects", []),
            "locations": self.task.get("locations", []),
        }

        return (
            "You are an LLM planner for a symbolic heterogeneous multi-robot behavior tree prototype.\n"
            "Output ONLY valid JSON. Do not use markdown. Do not include commentary.\n\n"
            "Task:\n"
            f"{json.dumps(task_payload, indent=2)}\n\n"
            "Robot capability library:\n"
            f"{json.dumps(self.capabilities, indent=2)}\n\n"
            "Required JSON schema:\n"
            "{\n"
            f'  "task_id": "{self.task.get("task_id", "task_id")}",\n'
            '  "task_graph": [\n'
            "    {\n"
            '      "id": "t1",\n'
            '      "action": "navigate",\n'
            '      "parameters": ["shelf"],\n'
            '      "depends_on": []\n'
            "    }\n"
            "  ],\n"
            '  "assignments": [\n'
            "    {\n"
            '      "task_id": "t1",\n'
            '      "robot": "go2_z1",\n'
            '      "action": "navigate",\n'
            '      "parameters": ["shelf"]\n'
            "    }\n"
            "  ],\n"
            '  "synchronization": [\n'
            "    {\n"
            '      "id": "sync_object_at_table",\n'
            '      "producer": "go2_z1",\n'
            '      "consumer": "franka",\n'
            '      "condition": "object_at(object, packing_table)",\n'
            '      "producer_task": "t4",\n'
            '      "consumer_task": "t6"\n'
            "    }\n"
            "  ],\n"
            '  "behavior_trees": {\n'
            '    "go2_z1": {\n'
            '      "type": "Sequence",\n'
            '      "children": []\n'
            "    },\n"
            '    "franka": {\n'
            '      "type": "Sequence",\n'
            '      "children": []\n'
            "    }\n"
            "  }\n"
            "}\n\n"
            "Constraints:\n"
            "1. Assign actions only to robots that have the matching capability.\n"
            "2. Include explicit synchronization for inter-robot dependencies.\n"
            "3. Franka must wait for object_at(object, packing_table) before insert(object, box).\n"
            "4. Go2-Z1 must wait for box_closed(box) before delivering the box.\n"
            "5. Preserve task dependency IDs consistently across task_graph, assignments, and synchronization.\n"
            "6. Output JSON only, no markdown.\n"
            f"{self._task_specific_prompt_constraints()}"
        )

    def _openai_plan(self, api_key: str) -> dict:
        url = _openai_chat_completions_url()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You produce strict JSON plans for symbolic multi-robot behavior tree planning.",
                },
                {"role": "user", "content": self.build_prompt()},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(_http_error_summary(error.code, detail)) from error

        api_result = json.loads(response_body)
        content = api_result["choices"][0]["message"]["content"]
        plan = extract_json(content)
        plan.setdefault("task_id", self.task.get("task_id"))
        plan.setdefault("instruction", self.task.get("instruction"))
        plan.setdefault("_llm_metadata", {})
        plan["_llm_metadata"].update({"mode_used": self.mode_used, "model": self.model})
        return plan

    def _mock_good_plan(self) -> dict:
        plan = RuleBasedPlanner(self.task, self.capabilities).generate_plan()
        self._add_assignment_action_fields(plan)
        plan["_llm_metadata"] = {"mode_used": self.mode_used, "model": "mock"}
        return plan

    def _mock_bad_plan(self) -> dict:
        plan = RuleBasedPlanner(self.task, self.capabilities).generate_plan()
        self._add_assignment_action_fields(plan)
        plan["synchronization"] = []

        for assignment in plan.get("assignments", []):
            if assignment.get("task_id") == "t7":
                assignment["robot"] = "go2_z1"

        franka_tree = plan.get("behavior_trees", {}).get("franka", {})
        franka_tree["children"] = [
            child
            for child in franka_tree.get("children", [])
            if not (
                child.get("type") == "Condition"
                and child.get("name") == "object_at"
                and child.get("parameters") == ["object", "packing_table"]
            )
        ]

        plan["_llm_metadata"] = {"mode_used": self.mode_used, "model": "mock"}
        if self.last_error:
            plan["_llm_metadata"]["fallback_reason"] = self.last_error
        return plan

    def _add_assignment_action_fields(self, plan: dict) -> None:
        task_by_id = {node.get("id"): node for node in plan.get("task_graph", [])}
        for assignment in plan.get("assignments", []):
            task = task_by_id.get(assignment.get("task_id"), {})
            assignment.setdefault("action", task.get("action"))
            assignment.setdefault("parameters", task.get("parameters", []))

    def _task_specific_prompt_constraints(self) -> str:
        if self.task.get("task_id") != "gear_assembly":
            return ""

        return (
            "\nGear assembly task-specific constraints:\n"
            "- There are three robots: go2_z1, franka1, franka2.\n"
            "- go2_z1 handles drawer operations, gear tray transport, and screwdriver transport.\n"
            "- franka1 holds and stabilizes the gearbase.\n"
            "- franka2 picks the gear, mounts the gear, picks the screwdriver, and fastens the screw.\n"
            "- Use place_tray(gear_tray, parts_zone), not carry(...), to make the tray available.\n"
            "- Use place_tool(screwdriver, tool_zone), not carry(...), to make the screwdriver available.\n"
            "- franka2 must wait for Condition tray_at gear_tray parts_zone before pick_gear gear.\n"
            "- franka2 must wait for Condition gearbase_stable gearbase before mount_gear gear shaft.\n"
            "- franka2 must wait for Condition tool_at screwdriver tool_zone before pick_tool screwdriver.\n"
            "- go2_z1 must wait for Condition screw_fastened gearbase before return_tool screwdriver parts_drawer and close_drawer parts_drawer.\n"
            "- Behavior tree children must be full node dictionaries, not task IDs or synchronization IDs.\n"
            "- The task graph should include t1 through t13 for the gear assembly sequence.\n"
            "- Output JSON only.\n"
        )


def _strip_markdown_fence(text: str) -> str:
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _openai_chat_completions_url() -> str:
    explicit_url = os.environ.get("OPENAI_API_URL")
    if explicit_url:
        return explicit_url

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _load_dotenv_if_present() -> None:
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidate_paths:
        if path.exists():
            _load_dotenv_file(path)
            return


def _load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _http_error_summary(status_code: int, detail: str) -> str:
    try:
        payload = json.loads(detail)
        error = payload.get("error", {})
        code = error.get("code") or "unknown_error"
        error_type = error.get("type") or "unknown_type"
        return f"HTTP {status_code}: {code} ({error_type})"
    except json.JSONDecodeError:
        return f"HTTP {status_code}"


def _safe_error_message(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    message = re.sub(r"sk-proj-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    return message
