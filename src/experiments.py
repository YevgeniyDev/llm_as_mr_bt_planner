from __future__ import annotations

from copy import deepcopy


def build_experiment_plans(base_plan: dict) -> dict:
    """Create deterministic ablation plans from a known-good base plan."""
    if base_plan.get("task_id") == "gear_assembly":
        return _build_gear_experiment_plans(base_plan)

    return {
        "full_pipeline": deepcopy(base_plan),
        "no_synchronization": _without_synchronization(base_plan),
        "wrong_assignment": _with_wrong_assignment(base_plan),
        "missing_handoff_condition": _without_franka_object_condition(base_plan),
        "bad_execution_order": _with_bad_franka_order(base_plan),
    }


def _build_gear_experiment_plans(base_plan: dict) -> dict:
    return {
        "full_pipeline": deepcopy(base_plan),
        "no_synchronization": _without_synchronization(base_plan),
        "wrong_assignment": _with_wrong_gear_assignment(base_plan),
        "missing_handoff_condition": _without_franka2_tool_condition(base_plan),
        "bad_execution_order": _with_bad_franka2_order(base_plan),
    }


def _without_synchronization(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    plan["synchronization"] = []
    return plan


def _with_wrong_assignment(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    franka_task_ids = {
        node.get("id")
        for node in plan.get("task_graph", [])
        if node.get("action") in {"open_box", "insert", "close_box"}
    }
    for assignment in plan.get("assignments", []):
        if assignment.get("task_id") in franka_task_ids:
            assignment["robot"] = "go2_z1"
    return plan


def _with_wrong_gear_assignment(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    franka2_task_ids = {
        node.get("id")
        for node in plan.get("task_graph", [])
        if node.get("action") in {"pick_gear", "mount_gear", "fasten_screw"}
    }
    for assignment in plan.get("assignments", []):
        if assignment.get("task_id") in franka2_task_ids:
            assignment["robot"] = "go2_z1"
    return plan


def _without_franka_object_condition(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    franka_tree = plan.get("behavior_trees", {}).get("franka", {})
    franka_tree["children"] = [
        child
        for child in franka_tree.get("children", [])
        if not _is_condition(child, "object_at", ["object", "packing_table"])
    ]
    return plan


def _without_franka2_tool_condition(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    franka2_tree = plan.get("behavior_trees", {}).get("franka2", {})
    franka2_tree["children"] = [
        child
        for child in franka2_tree.get("children", [])
        if not _is_condition(child, "tool_at", ["screwdriver", "tool_zone"])
    ]
    return plan


def _with_bad_franka_order(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    franka_tree = plan.get("behavior_trees", {}).get("franka", {})
    franka_tree["children"] = [
        {"type": "Condition", "name": "object_at", "parameters": ["object", "packing_table"]},
        {"type": "Action", "name": "insert", "parameters": ["object", "box"]},
        {"type": "Action", "name": "open_box", "parameters": ["box"]},
        {"type": "Action", "name": "close_box", "parameters": ["box"]},
    ]
    return plan


def _with_bad_franka2_order(base_plan: dict) -> dict:
    plan = deepcopy(base_plan)
    franka2_tree = plan.get("behavior_trees", {}).get("franka2", {})
    franka2_tree["children"] = [
        {"type": "Condition", "name": "tray_at", "parameters": ["gear_tray", "parts_zone"]},
        {"type": "Condition", "name": "gearbase_stable", "parameters": ["gearbase"]},
        {"type": "Action", "name": "mount_gear", "parameters": ["gear", "shaft"]},
        {"type": "Action", "name": "pick_gear", "parameters": ["gear"]},
        {"type": "Condition", "name": "tool_at", "parameters": ["screwdriver", "tool_zone"]},
        {"type": "Action", "name": "pick_tool", "parameters": ["screwdriver"]},
        {"type": "Action", "name": "fasten_screw", "parameters": ["gearbase", "screwdriver"]},
    ]
    return plan


def _is_condition(node: dict, name: str, parameters: list[str]) -> bool:
    return (
        node.get("type") == "Condition"
        and node.get("name") == name
        and node.get("parameters") == parameters
    )
