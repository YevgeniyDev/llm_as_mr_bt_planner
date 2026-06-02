from __future__ import annotations


class BTValidator:
    """Validate symbolic task assignments, synchronization, and BT syntax."""

    REQUIRED_FIELDS = ["task_graph", "assignments", "synchronization", "behavior_trees"]

    def __init__(self, plan: dict, capabilities: dict):
        self.plan = plan
        self.capabilities = capabilities
        self.errors: list[dict] = []
        self.robot_capabilities = self._build_robot_capability_index()

    def validate(self) -> dict:
        self.errors = []
        self._validate_required_fields()

        if self.errors:
            return self._result()

        self._validate_assignments()
        self._validate_behavior_trees()
        self._validate_synchronization()
        self._validate_bt_synchronization_consistency()
        self._validate_bt_ordering()
        return self._result()

    def _result(self) -> dict:
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
        }

    def _add_error(self, error_type: str, message: str) -> None:
        self.errors.append({"type": error_type, "message": message})

    def _build_robot_capability_index(self) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        for robot in self.capabilities.get("robots", []):
            index[robot.get("id", "")] = {
                capability.get("name", "")
                for capability in robot.get("capabilities", [])
                if capability.get("name")
            }
        return index

    def _validate_required_fields(self) -> None:
        for field in self.REQUIRED_FIELDS:
            if field not in self.plan:
                self._add_error("missing_field", f"Plan is missing required top-level field '{field}'.")

    def _validate_assignments(self) -> None:
        task_by_id = {node.get("id"): node for node in self.plan.get("task_graph", [])}

        for assignment in self.plan.get("assignments", []):
            task_id = assignment.get("task_id")
            robot = assignment.get("robot")

            if robot not in self.robot_capabilities:
                self._add_error(
                    "unknown_robot",
                    f"Task '{task_id}' is assigned to unknown robot '{robot}'.",
                )
                continue

            task_node = task_by_id.get(task_id)
            if not task_node:
                self._add_error(
                    "unknown_task",
                    f"Assignment references unknown task graph node '{task_id}'.",
                )
                continue

            action = task_node.get("action")
            if action not in self.robot_capabilities[robot]:
                self._add_error(
                    "invalid_capability",
                    f"Robot '{robot}' cannot execute action '{action}' assigned by task '{task_id}'.",
                )

    def _validate_behavior_trees(self) -> None:
        for robot, tree in self.plan.get("behavior_trees", {}).items():
            if robot not in self.robot_capabilities:
                self._add_error(
                    "unknown_robot",
                    f"Behavior tree is defined for unknown robot '{robot}'.",
                )
                continue
            self._validate_bt_node(tree, robot, path=f"behavior_trees.{robot}")

    def _validate_bt_node(self, node: dict, robot: str, path: str) -> None:
        if not isinstance(node, dict):
            self._add_error("invalid_bt_syntax", f"{path} must be a dictionary.")
            return

        node_type = node.get("type")
        if not node_type:
            self._add_error("invalid_bt_syntax", f"{path} is missing required field 'type'.")
            return

        if node_type == "Action":
            name = node.get("name")
            parameters = node.get("parameters")
            if not name:
                self._add_error("invalid_bt_syntax", f"{path} Action node is missing 'name'.")
            if not isinstance(parameters, list):
                self._add_error("invalid_bt_syntax", f"{path} Action node must have list field 'parameters'.")
            if name and name not in self.robot_capabilities.get(robot, set()):
                self._add_error(
                    "invalid_bt_action",
                    f"Robot '{robot}' cannot execute BT action '{name}' at {path}.",
                )
            return

        if node_type == "Condition":
            if not node.get("name"):
                self._add_error("invalid_bt_syntax", f"{path} Condition node is missing 'name'.")
            if not isinstance(node.get("parameters"), list):
                self._add_error("invalid_bt_syntax", f"{path} Condition node must have list field 'parameters'.")
            return

        if node_type in {"Sequence", "Fallback"}:
            children = node.get("children")
            if not isinstance(children, list):
                self._add_error("invalid_bt_syntax", f"{path} {node_type} node must have list field 'children'.")
                return
            for index, child in enumerate(children):
                self._validate_bt_node(child, robot, path=f"{path}.children[{index}]")
            return

        self._add_error("invalid_bt_syntax", f"{path} has unsupported BT node type '{node_type}'.")

    def _validate_synchronization(self) -> None:
        if self._is_gear_assembly_plan():
            self._validate_gear_synchronization()
            return

        conditions = {
            sync.get("condition")
            for sync in self.plan.get("synchronization", [])
            if sync.get("condition")
        }
        assignments = {item.get("task_id"): item.get("robot") for item in self.plan.get("assignments", [])}

        franka_inserts = False
        go2_uses_closed_box = False
        for node in self.plan.get("task_graph", []):
            task_id = node.get("id")
            action = node.get("action")
            parameters = node.get("parameters", [])
            robot = assignments.get(task_id)

            if robot == "franka" and action == "insert" and parameters == ["object", "box"]:
                franka_inserts = True

            if robot == "go2_z1" and action in {"pick", "deliver"} and "box" in parameters:
                go2_uses_closed_box = True

        if franka_inserts and "object_at(object, packing_table)" not in conditions:
            self._add_error(
                "missing_synchronization",
                "Franka inserts the object into the box, but synchronization condition "
                "'object_at(object, packing_table)' is missing.",
            )

        if go2_uses_closed_box and "box_closed(box)" not in conditions:
            self._add_error(
                "missing_synchronization",
                "Go2-Z1 handles the box after Franka closes it, but synchronization condition "
                "'box_closed(box)' is missing.",
            )

    def _validate_bt_synchronization_consistency(self) -> None:
        if self._is_gear_assembly_plan():
            self._validate_gear_bt_synchronization_consistency()
            return

        for sync in self.plan.get("synchronization", []):
            condition = sync.get("condition")
            consumer = sync.get("consumer")
            nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get(consumer, {}))

            if condition == "object_at(object, packing_table)":
                condition_index = self._find_condition_index(nodes, "object_at", ["object", "packing_table"])
                insert_index = self._find_action_index(nodes, "insert", ["object", "box"])
                if insert_index is not None and not self._index_before(condition_index, insert_index):
                    self._add_error(
                        "sync_consistency_error",
                        "Franka consumes synchronization condition 'object_at(object, packing_table)' "
                        "for insert(object, box), but its BT does not wait for that condition before insert.",
                    )

            if condition == "box_closed(box)":
                condition_index = self._find_condition_index(nodes, "box_closed", ["box"])
                first_box_action_index = self._find_first_box_action_index(nodes)
                if first_box_action_index is not None and not self._index_before(condition_index, first_box_action_index):
                    self._add_error(
                        "sync_consistency_error",
                        "Go2-Z1 consumes synchronization condition 'box_closed(box)', but its BT does not "
                        "wait for that condition before pick(box) or deliver(box, dropoff_zone).",
                    )

    def _validate_bt_ordering(self) -> None:
        if self._is_gear_assembly_plan():
            self._validate_gear_bt_ordering()
            return

        franka_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("franka", {}))
        open_index = self._find_action_index(franka_nodes, "open_box", ["box"])
        insert_index = self._find_action_index(franka_nodes, "insert", ["object", "box"])
        close_index = self._find_action_index(franka_nodes, "close_box", ["box"])

        if open_index is not None and insert_index is not None and open_index > insert_index:
            self._add_error(
                "bt_order_error",
                "Franka BT must execute open_box(box) before insert(object, box).",
            )

        if insert_index is not None and close_index is not None and insert_index > close_index:
            self._add_error(
                "bt_order_error",
                "Franka BT must execute insert(object, box) before close_box(box).",
            )

        go2_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("go2_z1", {}))
        place_index = self._find_action_index(go2_nodes, "place", ["object", "packing_table"])
        if insert_index is not None and place_index is None:
            self._add_error(
                "bt_order_error",
                "Go2-Z1 BT must include place(object, packing_table) before Franka can insert the object.",
            )

        if not self._has_task_graph_dependency("t4", "t6"):
            self._add_error(
                "bt_order_error",
                "Task graph must encode the handoff dependency from place(object, packing_table) to insert(object, box).",
            )

        box_closed_index = self._find_condition_index(go2_nodes, "box_closed", ["box"])
        first_box_action_index = self._find_first_box_action_index(go2_nodes)
        if first_box_action_index is not None and not self._index_before(box_closed_index, first_box_action_index):
            self._add_error(
                "bt_order_error",
                "Go2-Z1 BT must wait for box_closed(box) before pick(box) or deliver(box, dropoff_zone).",
            )

    def _flatten_tree(self, node: dict) -> list[dict]:
        if not isinstance(node, dict):
            return []

        node_type = node.get("type")
        if node_type in {"Sequence", "Fallback"}:
            flattened = []
            for child in node.get("children", []):
                flattened.extend(self._flatten_tree(child))
            return flattened

        return [node]

    def _find_condition_index(self, nodes: list[dict], name: str, parameters: list[str]) -> int | None:
        for index, node in enumerate(nodes):
            if node.get("type") == "Condition" and node.get("name") == name and node.get("parameters") == parameters:
                return index
        return None

    def _find_action_index(self, nodes: list[dict], name: str, parameters: list[str]) -> int | None:
        for index, node in enumerate(nodes):
            if node.get("type") == "Action" and node.get("name") == name and node.get("parameters") == parameters:
                return index
        return None

    def _find_first_box_action_index(self, nodes: list[dict]) -> int | None:
        for index, node in enumerate(nodes):
            if node.get("type") != "Action":
                continue
            if node.get("name") in {"pick", "deliver"} and "box" in node.get("parameters", []):
                return index
        return None

    def _index_before(self, maybe_before: int | None, after: int) -> bool:
        return maybe_before is not None and maybe_before < after

    def _has_task_graph_dependency(self, producer_task: str, consumer_task: str) -> bool:
        task_by_id = {node.get("id"): node for node in self.plan.get("task_graph", [])}
        consumer_node = task_by_id.get(consumer_task, {})
        return producer_task in consumer_node.get("depends_on", [])

    def _is_gear_assembly_plan(self) -> bool:
        if self.plan.get("task_id") == "gear_assembly":
            return True
        actions = {node.get("action") for node in self.plan.get("task_graph", []) if isinstance(node, dict)}
        return bool(actions & {"open_drawer", "mount_gear", "fasten_screw"})

    def _validate_gear_synchronization(self) -> None:
        conditions = {
            sync.get("condition")
            for sync in self.plan.get("synchronization", [])
            if sync.get("condition")
        }
        required_conditions = [
            (
                "tray_at(gear_tray, parts_zone)",
                "Franka2 picks or mounts the gear, but synchronization condition "
                "'tray_at(gear_tray, parts_zone)' is missing.",
            ),
            (
                "gearbase_stable(gearbase)",
                "Franka2 mounts the gear, but synchronization condition "
                "'gearbase_stable(gearbase)' is missing.",
            ),
            (
                "tool_at(screwdriver, tool_zone)",
                "Franka2 uses the screwdriver, but synchronization condition "
                "'tool_at(screwdriver, tool_zone)' is missing.",
            ),
            (
                "screw_fastened(gearbase)",
                "Go2-Z1 returns the screwdriver after fastening, but synchronization condition "
                "'screw_fastened(gearbase)' is missing.",
            ),
        ]
        for condition, message in required_conditions:
            if condition not in conditions:
                self._add_error("missing_synchronization", message)

    def _validate_gear_bt_synchronization_consistency(self) -> None:
        franka2_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("franka2", {}))
        go2_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("go2_z1", {}))

        self._require_condition_before_action(
            nodes=franka2_nodes,
            condition_name="tray_at",
            condition_parameters=["gear_tray", "parts_zone"],
            action_name="pick_gear",
            action_parameters=["gear"],
            message="Franka2 must wait for tray_at(gear_tray, parts_zone) before pick_gear(gear).",
        )
        self._require_condition_before_action(
            nodes=franka2_nodes,
            condition_name="gearbase_stable",
            condition_parameters=["gearbase"],
            action_name="mount_gear",
            action_parameters=["gear", "shaft"],
            message="Franka2 must wait for gearbase_stable(gearbase) before mount_gear(gear, shaft).",
        )
        self._require_condition_before_action(
            nodes=franka2_nodes,
            condition_name="tool_at",
            condition_parameters=["screwdriver", "tool_zone"],
            action_name="pick_tool",
            action_parameters=["screwdriver"],
            message="Franka2 must wait for tool_at(screwdriver, tool_zone) before pick_tool(screwdriver).",
        )
        self._require_condition_before_action(
            nodes=go2_nodes,
            condition_name="screw_fastened",
            condition_parameters=["gearbase"],
            action_name="return_tool",
            action_parameters=["screwdriver", "parts_drawer"],
            message="Go2-Z1 must wait for screw_fastened(gearbase) before return_tool(screwdriver, parts_drawer).",
        )

    def _validate_gear_bt_ordering(self) -> None:
        franka1_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("franka1", {}))
        franka2_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("franka2", {}))
        go2_nodes = self._flatten_tree(self.plan.get("behavior_trees", {}).get("go2_z1", {}))

        hold_index = self._find_action_index(franka1_nodes, "hold_gearbase", ["gearbase"])
        stabilize_index = self._find_action_index(franka1_nodes, "stabilize_gearbase", ["gearbase"])
        if hold_index is not None and stabilize_index is not None and hold_index > stabilize_index:
            self._add_error("bt_order_error", "Franka1 must hold the gearbase before stabilizing it.")

        pick_gear_index = self._find_action_index(franka2_nodes, "pick_gear", ["gear"])
        mount_index = self._find_action_index(franka2_nodes, "mount_gear", ["gear", "shaft"])
        pick_tool_index = self._find_action_index(franka2_nodes, "pick_tool", ["screwdriver"])
        fasten_index = self._find_action_index(franka2_nodes, "fasten_screw", ["gearbase", "screwdriver"])
        if pick_gear_index is not None and mount_index is not None and pick_gear_index > mount_index:
            self._add_error("bt_order_error", "Franka2 must pick_gear(gear) before mount_gear(gear, shaft).")
        if mount_index is not None and fasten_index is not None and mount_index > fasten_index:
            self._add_error("bt_order_error", "Franka2 must mount_gear(gear, shaft) before fasten_screw(gearbase, screwdriver).")
        if pick_tool_index is not None and fasten_index is not None and pick_tool_index > fasten_index:
            self._add_error("bt_order_error", "Franka2 must pick_tool(screwdriver) before fasten_screw(gearbase, screwdriver).")

        return_index = self._find_action_index(go2_nodes, "return_tool", ["screwdriver", "parts_drawer"])
        close_index = self._find_action_index(go2_nodes, "close_drawer", ["parts_drawer"])
        if return_index is not None and close_index is not None and return_index > close_index:
            self._add_error("bt_order_error", "Go2-Z1 must return the screwdriver before closing the drawer.")

        required_dependencies = [
            ("t3", "t8", "Task graph must encode tray handoff dependency t3 -> t8."),
            ("t7", "t9", "Task graph must encode gearbase stabilization dependency t7 -> t9."),
            ("t5", "t10", "Task graph must encode screwdriver handoff dependency t5 -> t10."),
            ("t11", "t12", "Task graph must encode screw-fastened handoff dependency t11 -> t12."),
        ]
        for producer_task, consumer_task, message in required_dependencies:
            if not self._has_task_graph_dependency(producer_task, consumer_task):
                self._add_error("bt_order_error", message)

    def _require_condition_before_action(
        self,
        nodes: list[dict],
        condition_name: str,
        condition_parameters: list[str],
        action_name: str,
        action_parameters: list[str],
        message: str,
    ) -> None:
        action_index = self._find_action_index(nodes, action_name, action_parameters)
        if action_index is None:
            return
        condition_index = self._find_condition_index(nodes, condition_name, condition_parameters)
        if not self._index_before(condition_index, action_index):
            self._add_error("sync_consistency_error", message)
