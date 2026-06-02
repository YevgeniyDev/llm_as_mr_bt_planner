from __future__ import annotations

import re


class SymbolicSimulator:
    """Small symbolic BT interpreter for the packaging-delivery prototype."""

    def __init__(self, initial_state: list[str], behavior_trees: dict, capabilities: dict):
        self.state = set(initial_state)
        self.behavior_trees = behavior_trees
        self.capability_index = self._build_capability_index(capabilities)
        self.trace: list[dict] = []
        self.errors: list[dict] = []

    def run(self) -> dict:
        if self._is_gear_assembly_run():
            return self._run_gear_assembly()
        return self._run_packaging()

    def _run_packaging(self) -> dict:
        go2_tree = self.behavior_trees.get("go2_z1")
        franka_tree = self.behavior_trees.get("franka")

        if not go2_tree:
            self._add_error("missing_bt", "Missing behavior tree for robot 'go2_z1'.")
        if not franka_tree:
            self._add_error("missing_bt", "Missing behavior tree for robot 'franka'.")
        if self.errors:
            return self._result(False)

        pause_index = self._find_sync_condition_index(go2_tree, "box_closed", ["box"])
        if pause_index is None:
            go2_before_franka = self._execute_node(go2_tree, "go2_z1")
            if not go2_before_franka:
                return self._result(False)
        else:
            before_sync = {
                "type": "Sequence",
                "children": go2_tree.get("children", [])[:pause_index],
            }
            if not self._execute_node(before_sync, "go2_z1"):
                return self._result(False)
            self.trace.append(
                {
                    "robot": "go2_z1",
                    "event": "pause",
                    "condition": "box_closed(box)",
                    "reason": "waiting for franka handoff",
                }
            )

        if not self._execute_node(franka_tree, "franka"):
            return self._result(False)

        if pause_index is not None:
            after_sync = {
                "type": "Sequence",
                "children": go2_tree.get("children", [])[pause_index:],
            }
            if not self._execute_node(after_sync, "go2_z1"):
                return self._result(False)

        return self._result(True)

    def _run_gear_assembly(self) -> dict:
        go2_tree = self.behavior_trees.get("go2_z1")
        franka1_tree = self.behavior_trees.get("franka1")
        franka2_tree = self.behavior_trees.get("franka2")

        if not go2_tree:
            self._add_error("missing_bt", "Missing behavior tree for robot 'go2_z1'.")
        if not franka1_tree:
            self._add_error("missing_bt", "Missing behavior tree for robot 'franka1'.")
        if not franka2_tree:
            self._add_error("missing_bt", "Missing behavior tree for robot 'franka2'.")
        if self.errors:
            return self._result(False)

        pause_index = self._find_sync_condition_index(go2_tree, "screw_fastened", ["gearbase"])
        if pause_index is None:
            if not self._execute_node(go2_tree, "go2_z1"):
                return self._result(False)
        else:
            before_sync = {
                "type": "Sequence",
                "children": go2_tree.get("children", [])[:pause_index],
            }
            if not self._execute_node(before_sync, "go2_z1"):
                return self._result(False)
            self.trace.append(
                {
                    "robot": "go2_z1",
                    "event": "pause",
                    "condition": "screw_fastened(gearbase)",
                    "reason": "waiting for franka2 handoff",
                }
            )

        if not self._execute_node(franka1_tree, "franka1"):
            return self._result(False)

        if not self._execute_node(franka2_tree, "franka2"):
            return self._result(False)

        if pause_index is not None:
            after_sync = {
                "type": "Sequence",
                "children": go2_tree.get("children", [])[pause_index:],
            }
            if not self._execute_node(after_sync, "go2_z1"):
                return self._result(False)

        return self._result(True)

    def _result(self, success: bool) -> dict:
        return {
            "success": success and not self.errors,
            "final_state": sorted(self.state),
            "trace": self.trace,
            "errors": self.errors,
        }

    def _add_error(self, error_type: str, message: str) -> None:
        self.errors.append({"type": error_type, "message": message})

    def _build_capability_index(self, capabilities: dict) -> dict[str, dict[str, dict]]:
        index: dict[str, dict[str, dict]] = {}
        for robot in capabilities.get("robots", []):
            robot_id = robot.get("id")
            if not robot_id:
                continue
            index[robot_id] = {
                capability.get("name"): capability
                for capability in robot.get("capabilities", [])
                if capability.get("name")
            }
        return index

    def _is_gear_assembly_run(self) -> bool:
        return "franka1" in self.behavior_trees or "franka2" in self.behavior_trees

    def _find_sync_condition_index(self, tree: dict, name: str, parameters: list[str]) -> int | None:
        if tree.get("type") != "Sequence":
            return None
        for index, child in enumerate(tree.get("children", [])):
            if child.get("type") == "Condition" and child.get("name") == name and child.get("parameters") == parameters:
                return index
        return None

    def _execute_node(self, node: dict, robot: str) -> bool:
        node_type = node.get("type")

        if node_type == "Sequence":
            for child in node.get("children", []):
                if not self._execute_node(child, robot):
                    return False
            return True

        if node_type == "Fallback":
            branch_errors: list[dict] = []
            for child in node.get("children", []):
                state_snapshot = set(self.state)
                trace_length = len(self.trace)
                error_length = len(self.errors)
                if self._execute_node(child, robot):
                    return True

                branch_errors = self.errors[error_length:]
                self.state = state_snapshot
                del self.trace[trace_length:]
                del self.errors[error_length:]

            details = "; ".join(error.get("message", "") for error in branch_errors)
            message = f"All fallback children failed for robot '{robot}'."
            if details:
                message = f"{message} Last failure: {details}"
            self._add_error("fallback_failed", message)
            return False

        if node_type == "Condition":
            return self._execute_condition(node, robot)

        if node_type == "Action":
            return self._execute_action(node, robot)

        self._add_error("invalid_bt_node", f"Unsupported BT node type '{node_type}' for robot '{robot}'.")
        return False

    def _execute_condition(self, node: dict, robot: str) -> bool:
        predicate = self._format_predicate(node.get("name", ""), node.get("parameters", []))
        if self._predicate_satisfied(predicate):
            self.trace.append({"robot": robot, "event": "condition_success", "condition": predicate})
            return True

        self.trace.append({"robot": robot, "event": "condition_failure", "condition": predicate})
        self._add_error("condition_failed", f"Condition '{predicate}' failed for robot '{robot}'.")
        return False

    def _execute_action(self, node: dict, robot: str) -> bool:
        action_name = node.get("name")
        capability = self.capability_index.get(robot, {}).get(action_name)
        if not capability:
            self._add_error("unknown_action", f"Robot '{robot}' has no capability '{action_name}'.")
            return False

        bindings = self._bind_parameters(capability.get("parameters", []), node.get("parameters", []))
        preconditions = [
            self._substitute(predicate, bindings)
            for predicate in capability.get("preconditions", [])
        ]

        missing = [
            predicate
            for predicate in preconditions
            if not self._predicate_satisfied(predicate)
        ]
        if missing:
            self._add_error(
                "precondition_failed",
                f"Action '{action_name}' for robot '{robot}' failed missing preconditions: {missing}.",
            )
            self.trace.append(
                {
                    "robot": robot,
                    "event": "action_failure",
                    "action": action_name,
                    "parameters": node.get("parameters", []),
                    "missing_preconditions": missing,
                }
            )
            return False

        effects = [
            self._substitute(predicate, bindings)
            for predicate in capability.get("effects", [])
        ]
        for effect in effects:
            self._apply_effect(effect)

        self.trace.append(
            {
                "robot": robot,
                "event": "action_success",
                "action": action_name,
                "parameters": node.get("parameters", []),
                "effects": effects,
            }
        )
        return True

    def _bind_parameters(self, formal_parameters: list[str], actual_parameters: list[str]) -> dict[str, str]:
        return dict(zip(formal_parameters, actual_parameters))

    def _substitute(self, predicate: str, bindings: dict[str, str]) -> str:
        result = predicate
        for variable, value in bindings.items():
            result = re.sub(rf"\b{re.escape(variable)}\b", value, result)
        return result

    def _predicate_satisfied(self, predicate: str) -> bool:
        if predicate in self.state:
            return True

        name, args = self._parse_predicate(predicate)
        if name == "robot_near" and len(args) == 2:
            return self._robot_near(args[0], args[1])

        return False

    def _robot_near(self, robot: str, entity: str) -> bool:
        robot_locations = {
            args[1]
            for predicate in self.state
            for name, args in [self._parse_predicate(predicate)]
            if name == "robot_at" and len(args) == 2 and args[0] == robot
        }
        entity_locations = {
            args[1]
            for predicate in self.state
            for name, args in [self._parse_predicate(predicate)]
            if name in {"object_at", "box_at"} and len(args) == 2 and args[0] == entity
        }
        return bool(robot_locations & entity_locations)

    def _apply_effect(self, effect: str) -> None:
        name, args = self._parse_predicate(effect)

        if name == "not_holding" and len(args) == 2:
            self.state.discard(self._format_predicate("holding", args))
            return

        if name == "not_object_at" and args:
            self._remove_matching("object_at", args[0])
            if args[0] == "box":
                self._remove_matching("box_at", args[0])
            return

        if name == "box_open" and len(args) == 1:
            self.state.discard(self._format_predicate("box_closed", args))

        if name == "box_closed" and len(args) == 1:
            self.state.discard(self._format_predicate("box_open", args))

        if name == "drawer_open" and len(args) == 1:
            self.state.discard(self._format_predicate("drawer_closed", args))

        if name == "drawer_closed" and len(args) == 1:
            self.state.discard(self._format_predicate("drawer_open", args))

        if name == "not_tool_at" and args:
            self._remove_matching("tool_at", args[0])
            return

        if name == "not_screwdriver_at" and args:
            self._remove_matching("screwdriver_at", args[0])
            return

        if name == "not_gear_tray_at" and args:
            self._remove_matching("gear_tray_at", args[0])
            self._remove_matching("tray_at", args[0])
            return

        if name == "robot_at" and len(args) == 2:
            self._remove_matching("robot_at", args[0])

        if name == "object_at" and len(args) == 2:
            self._remove_matching("object_at", args[0])
            if args[0] == "box":
                self._remove_matching("box_at", args[0])

        if name == "box_at" and len(args) == 2:
            self._remove_matching("box_at", args[0])

        if name in {"tool_at", "screwdriver_at", "gear_tray_at", "tray_at"} and len(args) == 2:
            self._remove_matching(name, args[0])

        self.state.add(effect)

    def _remove_matching(self, predicate_name: str, first_argument: str) -> None:
        to_remove = set()
        for predicate in self.state:
            name, args = self._parse_predicate(predicate)
            if name == predicate_name and args and args[0] == first_argument:
                to_remove.add(predicate)
        self.state.difference_update(to_remove)

    def _parse_predicate(self, predicate: str) -> tuple[str, list[str]]:
        if "(" not in predicate or not predicate.endswith(")"):
            return predicate, []
        name, raw_args = predicate.split("(", 1)
        args = [argument.strip() for argument in raw_args[:-1].split(",") if argument.strip()]
        return name.strip(), args

    def _format_predicate(self, name: str, args: list[str]) -> str:
        return f"{name}({', '.join(args)})"
