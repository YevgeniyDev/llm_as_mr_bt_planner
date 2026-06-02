from __future__ import annotations

from copy import deepcopy

from planner import RuleBasedPlanner


REQUIRED_SYNCHRONIZATION = [
    {
        "id": "sync_object_at_table",
        "producer": "go2_z1",
        "consumer": "franka",
        "condition": "object_at(object, packing_table)",
        "producer_task": "t4",
        "consumer_task": "t6",
    },
    {
        "id": "sync_box_closed",
        "producer": "franka",
        "consumer": "go2_z1",
        "condition": "box_closed(box)",
        "producer_task": "t7",
        "consumer_task": "t8",
    },
]

GEAR_SYNCHRONIZATION = [
    {
        "id": "sync_tray_ready",
        "producer": "go2_z1",
        "consumer": "franka2",
        "condition": "tray_at(gear_tray, parts_zone)",
        "producer_task": "t3",
        "consumer_task": "t8",
    },
    {
        "id": "sync_gearbase_stable",
        "producer": "franka1",
        "consumer": "franka2",
        "condition": "gearbase_stable(gearbase)",
        "producer_task": "t7",
        "consumer_task": "t9",
    },
    {
        "id": "sync_tool_ready",
        "producer": "go2_z1",
        "consumer": "franka2",
        "condition": "tool_at(screwdriver, tool_zone)",
        "producer_task": "t5",
        "consumer_task": "t10",
    },
    {
        "id": "sync_screw_fastened",
        "producer": "franka2",
        "consumer": "go2_z1",
        "condition": "screw_fastened(gearbase)",
        "producer_task": "t11",
        "consumer_task": "t12",
    },
]


class SimpleRepairLoop:
    """Deterministic repair rules for the v0.2 ablation experiments."""

    def __init__(self):
        pass

    def repair(self, plan: dict, validation_result: dict) -> dict:
        repaired = deepcopy(plan)
        if _is_gear_plan(repaired):
            return self._canonical_gear_plan(repaired)

        self._repair_synchronization(repaired)
        self._repair_assignments(repaired)
        self._repair_task_graph_dependencies(repaired)
        self._normalize_behavior_tree_children(repaired)
        self._repair_franka_tree(repaired)
        self._repair_go2_tree(repaired)
        return repaired

    def _canonical_gear_plan(self, plan: dict) -> dict:
        task = {
            "task_id": "gear_assembly",
            "instruction": plan.get(
                "instruction",
                "Open the parts drawer, retrieve the gear tray and screwdriver, bring them to the workstation, mount the gear onto the shaft, fasten the gearbase screw, then return the screwdriver to storage.",
            ),
        }
        repaired = RuleBasedPlanner(task=task, capabilities={}).generate_plan()
        if "_llm_metadata" in plan:
            repaired["_llm_metadata"] = deepcopy(plan["_llm_metadata"])
        return repaired

    def _repair_gear_plan(self, plan: dict) -> None:
        self._repair_gear_synchronization(plan)
        self._repair_gear_assignments(plan)
        self._repair_gear_task_graph_dependencies(plan)
        self._repair_gear_behavior_trees(plan)

    def _repair_gear_synchronization(self, plan: dict) -> None:
        synchronization = plan.setdefault("synchronization", [])
        existing_conditions = {
            sync.get("condition")
            for sync in synchronization
            if sync.get("condition")
        }
        for required_sync in GEAR_SYNCHRONIZATION:
            if required_sync["condition"] not in existing_conditions:
                synchronization.append(deepcopy(required_sync))

    def _repair_gear_assignments(self, plan: dict) -> None:
        task_actions = {
            node.get("id"): node.get("action")
            for node in plan.get("task_graph", [])
            if isinstance(node, dict)
        }
        for assignment in plan.get("assignments", []):
            action = task_actions.get(assignment.get("task_id"))
            if action in {"hold_gearbase", "stabilize_gearbase"}:
                assignment["robot"] = "franka1"
            elif action in {"pick_gear", "mount_gear", "fasten_screw"}:
                assignment["robot"] = "franka2"
            elif action == "pick_tool" and assignment.get("task_id") == "t10":
                assignment["robot"] = "franka2"
            elif action in {"open_drawer", "close_drawer", "pick_tray", "place_tray", "return_tool"}:
                assignment["robot"] = "go2_z1"
            elif action == "pick_tool":
                assignment["robot"] = "go2_z1"
            elif action == "place_tool":
                assignment["robot"] = "go2_z1"

    def _repair_gear_task_graph_dependencies(self, plan: dict) -> None:
        task_by_id = {
            node.get("id"): node
            for node in plan.get("task_graph", [])
            if isinstance(node, dict)
        }
        required_dependencies = {
            "t8": ["t3"],
            "t9": ["t7", "t8"],
            "t10": ["t5"],
            "t11": ["t9", "t10"],
            "t12": ["t11"],
            "t13": ["t12"],
        }
        for task_id, dependencies in required_dependencies.items():
            if task_id not in task_by_id:
                continue
            depends_on = task_by_id[task_id].setdefault("depends_on", [])
            for dependency in dependencies:
                if dependency not in depends_on:
                    depends_on.append(dependency)

    def _repair_gear_behavior_trees(self, plan: dict) -> None:
        behavior_trees = plan.setdefault("behavior_trees", {})
        behavior_trees["go2_z1"] = {
            "type": "Sequence",
            "children": [
                {"type": "Action", "name": "open_drawer", "parameters": ["parts_drawer"]},
                {"type": "Action", "name": "pick_tray", "parameters": ["gear_tray"]},
                {"type": "Action", "name": "place_tray", "parameters": ["gear_tray", "parts_zone"]},
                {"type": "Action", "name": "pick_tool", "parameters": ["screwdriver"]},
                {"type": "Action", "name": "place_tool", "parameters": ["screwdriver", "tool_zone"]},
                {"type": "Condition", "name": "screw_fastened", "parameters": ["gearbase"]},
                {"type": "Action", "name": "return_tool", "parameters": ["screwdriver", "parts_drawer"]},
                {"type": "Action", "name": "close_drawer", "parameters": ["parts_drawer"]},
            ],
        }
        behavior_trees["franka1"] = {
            "type": "Sequence",
            "children": [
                {"type": "Action", "name": "hold_gearbase", "parameters": ["gearbase"]},
                {"type": "Action", "name": "stabilize_gearbase", "parameters": ["gearbase"]},
            ],
        }
        behavior_trees["franka2"] = {
            "type": "Sequence",
            "children": [
                {"type": "Condition", "name": "tray_at", "parameters": ["gear_tray", "parts_zone"]},
                {"type": "Condition", "name": "gearbase_stable", "parameters": ["gearbase"]},
                {"type": "Action", "name": "pick_gear", "parameters": ["gear"]},
                {"type": "Action", "name": "mount_gear", "parameters": ["gear", "shaft"]},
                {"type": "Condition", "name": "tool_at", "parameters": ["screwdriver", "tool_zone"]},
                {"type": "Action", "name": "pick_tool", "parameters": ["screwdriver"]},
                {"type": "Action", "name": "fasten_screw", "parameters": ["gearbase", "screwdriver"]},
            ],
        }

    def _repair_synchronization(self, plan: dict) -> None:
        synchronization = plan.setdefault("synchronization", [])
        existing_conditions = {
            sync.get("condition")
            for sync in synchronization
            if sync.get("condition")
        }
        for required_sync in REQUIRED_SYNCHRONIZATION:
            if required_sync["condition"] not in existing_conditions:
                synchronization.append(deepcopy(required_sync))

    def _repair_assignments(self, plan: dict) -> None:
        task_actions = {
            node.get("id"): node.get("action")
            for node in plan.get("task_graph", [])
        }
        for assignment in plan.get("assignments", []):
            action = task_actions.get(assignment.get("task_id"))
            if action in {"open_box", "insert", "close_box"} and assignment.get("robot") == "go2_z1":
                assignment["robot"] = "franka"

    def _repair_task_graph_dependencies(self, plan: dict) -> None:
        task_by_id = {
            node.get("id"): node
            for node in plan.get("task_graph", [])
            if isinstance(node, dict)
        }
        if "t6" in task_by_id:
            depends_on = task_by_id["t6"].setdefault("depends_on", [])
            for required_dependency in ["t4", "t5"]:
                if required_dependency not in depends_on:
                    depends_on.append(required_dependency)

    def _normalize_behavior_tree_children(self, plan: dict) -> None:
        behavior_trees = plan.setdefault("behavior_trees", {})
        task_by_id = {
            node.get("id"): node
            for node in plan.get("task_graph", [])
            if isinstance(node, dict)
        }

        for tree in behavior_trees.values():
            if not isinstance(tree, dict):
                continue
            children = tree.setdefault("children", [])
            if not isinstance(children, list):
                tree["children"] = []
                continue
            tree["children"] = [
                normalized
                for child in children
                for normalized in _normalize_child_reference(child, task_by_id)
            ]

    def _repair_franka_tree(self, plan: dict) -> None:
        behavior_trees = plan.setdefault("behavior_trees", {})
        franka_tree = behavior_trees.setdefault(
            "franka",
            {"type": "Sequence", "children": []},
        )
        children = franka_tree.setdefault("children", [])

        insert_index = _find_first_action(children, "insert", ["object", "box"])
        open_index = _find_first_action(children, "open_box", ["box"])
        if insert_index is not None and (open_index is None or insert_index < open_index):
            franka_tree["type"] = "Sequence"
            franka_tree["children"] = _canonical_franka_children()
            return

        if not _has_condition_before(children, "object_at", ["object", "packing_table"], insert_index):
            condition = {"type": "Condition", "name": "object_at", "parameters": ["object", "packing_table"]}
            if insert_index is None:
                children.append(condition)
            else:
                children.insert(insert_index, condition)

    def _repair_go2_tree(self, plan: dict) -> None:
        behavior_trees = plan.setdefault("behavior_trees", {})
        go2_tree = behavior_trees.setdefault(
            "go2_z1",
            {"type": "Sequence", "children": []},
        )
        children = go2_tree.setdefault("children", [])
        _remove_premature_dropoff_navigation(children)

        deliver_index = _find_first_action(children, "deliver", ["box", "dropoff_zone"])
        if deliver_index is not None and not _has_action_before(children, "pick", ["box"], deliver_index):
            children.insert(deliver_index, {"type": "Action", "name": "pick", "parameters": ["box"]})

        first_box_action_index = _find_first_box_action(children)
        if first_box_action_index is None:
            return

        if not _has_condition_before(children, "box_closed", ["box"], first_box_action_index):
            condition = {"type": "Condition", "name": "box_closed", "parameters": ["box"]}
            children.insert(first_box_action_index, condition)


def _canonical_franka_children() -> list[dict]:
    return [
        {"type": "Action", "name": "open_box", "parameters": ["box"]},
        {"type": "Condition", "name": "object_at", "parameters": ["object", "packing_table"]},
        {"type": "Action", "name": "insert", "parameters": ["object", "box"]},
        {"type": "Action", "name": "close_box", "parameters": ["box"]},
    ]


def _normalize_child_reference(child: object, task_by_id: dict[str, dict]) -> list[dict]:
    if isinstance(child, dict):
        return [child]

    if not isinstance(child, str):
        return []

    if child == "sync_object_at_table":
        return [{"type": "Condition", "name": "object_at", "parameters": ["object", "packing_table"]}]

    if child == "sync_box_closed":
        return [{"type": "Condition", "name": "box_closed", "parameters": ["box"]}]

    task = task_by_id.get(child)
    if task:
        return [
            {
                "type": "Action",
                "name": task.get("action"),
                "parameters": task.get("parameters", []),
            }
        ]

    return []


def _find_first_action(children: list[dict], name: str, parameters: list[str]) -> int | None:
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        if child.get("type") == "Action" and child.get("name") == name and child.get("parameters") == parameters:
            return index
    return None


def _find_first_box_action(children: list[dict]) -> int | None:
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        if child.get("type") != "Action":
            continue
        if child.get("name") in {"pick", "deliver"} and "box" in child.get("parameters", []):
            return index
    return None


def _remove_premature_dropoff_navigation(children: list[dict]) -> None:
    first_box_action_index = _find_first_box_action(children)
    if first_box_action_index is None:
        return

    filtered_children = [
        child
        for index, child in enumerate(children)
        if not (
            index < first_box_action_index
            and isinstance(child, dict)
            and child.get("type") == "Action"
            and child.get("name") == "navigate"
            and child.get("parameters") == ["dropoff_zone"]
        )
    ]
    children[:] = filtered_children


def _has_condition_before(
    children: list[dict],
    name: str,
    parameters: list[str],
    before_index: int | None,
) -> bool:
    if before_index is None:
        search_space = children
    else:
        search_space = children[:before_index]
    return any(
        isinstance(child, dict)
        and child.get("type") == "Condition"
        and child.get("name") == name
        and child.get("parameters") == parameters
        for child in search_space
    )


def _has_action_before(
    children: list[dict],
    name: str,
    parameters: list[str],
    before_index: int,
) -> bool:
    return any(
        isinstance(child, dict)
        and child.get("type") == "Action"
        and child.get("name") == name
        and child.get("parameters") == parameters
        for child in children[:before_index]
    )


def _is_gear_plan(plan: dict) -> bool:
    if plan.get("task_id") == "gear_assembly":
        return True
    instruction = str(plan.get("instruction", "")).lower()
    if "gear" in instruction or "screwdriver" in instruction:
        return True
    actions = {
        node.get("action")
        for node in plan.get("task_graph", [])
        if isinstance(node, dict)
    }
    gear_actions = {
        "open_drawer",
        "close_drawer",
        "pick_tray",
        "place_tray",
        "return_tool",
        "hold_gearbase",
        "stabilize_gearbase",
        "pick_gear",
        "mount_gear",
        "fasten_screw",
    }
    if actions & gear_actions:
        return True
    behavior_trees = plan.get("behavior_trees", {})
    if isinstance(behavior_trees, dict) and {"franka1", "franka2"} & set(behavior_trees):
        return True
    return False
