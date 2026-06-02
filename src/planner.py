from __future__ import annotations


class RuleBasedPlanner:
    """Deterministic stand-in for the future LLM planning pipeline."""

    def __init__(self, task: dict, capabilities: dict):
        self.task = task
        self.capabilities = capabilities

    def generate_plan(self) -> dict:
        task_id = self.task.get("task_id")
        if task_id == "gear_assembly":
            return self._generate_gear_assembly_plan()
        return self._generate_packaging_plan()

    def _generate_packaging_plan(self) -> dict:
        task_graph = [
            {"id": "t1", "action": "navigate", "parameters": ["shelf"], "depends_on": []},
            {"id": "t2", "action": "pick", "parameters": ["object"], "depends_on": ["t1"]},
            {"id": "t3", "action": "carry", "parameters": ["object", "packing_table"], "depends_on": ["t2"]},
            {"id": "t4", "action": "place", "parameters": ["object", "packing_table"], "depends_on": ["t3"]},
            {"id": "t5", "action": "open_box", "parameters": ["box"], "depends_on": []},
            {"id": "t6", "action": "insert", "parameters": ["object", "box"], "depends_on": ["t4", "t5"]},
            {"id": "t7", "action": "close_box", "parameters": ["box"], "depends_on": ["t6"]},
            {"id": "t8", "action": "pick", "parameters": ["box"], "depends_on": ["t7"]},
            {"id": "t9", "action": "deliver", "parameters": ["box", "dropoff_zone"], "depends_on": ["t8"]},
        ]

        assignments = [
            {"task_id": "t1", "robot": "go2_z1"},
            {"task_id": "t2", "robot": "go2_z1"},
            {"task_id": "t3", "robot": "go2_z1"},
            {"task_id": "t4", "robot": "go2_z1"},
            {"task_id": "t5", "robot": "franka"},
            {"task_id": "t6", "robot": "franka"},
            {"task_id": "t7", "robot": "franka"},
            {"task_id": "t8", "robot": "go2_z1"},
            {"task_id": "t9", "robot": "go2_z1"},
        ]

        synchronization = [
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

        behavior_trees = {
            "go2_z1": {
                "type": "Sequence",
                "children": [
                    {"type": "Action", "name": "navigate", "parameters": ["shelf"]},
                    {"type": "Action", "name": "pick", "parameters": ["object"]},
                    {"type": "Action", "name": "carry", "parameters": ["object", "packing_table"]},
                    {"type": "Action", "name": "place", "parameters": ["object", "packing_table"]},
                    {"type": "Condition", "name": "box_closed", "parameters": ["box"]},
                    {"type": "Action", "name": "pick", "parameters": ["box"]},
                    {"type": "Action", "name": "deliver", "parameters": ["box", "dropoff_zone"]},
                ],
            },
            "franka": {
                "type": "Sequence",
                "children": [
                    {"type": "Action", "name": "open_box", "parameters": ["box"]},
                    {"type": "Condition", "name": "object_at", "parameters": ["object", "packing_table"]},
                    {"type": "Action", "name": "insert", "parameters": ["object", "box"]},
                    {"type": "Action", "name": "close_box", "parameters": ["box"]},
                ],
            },
        }

        return {
            "task_id": self.task.get("task_id"),
            "instruction": self.task.get("instruction"),
            "task_graph": task_graph,
            "assignments": assignments,
            "synchronization": synchronization,
            "behavior_trees": behavior_trees,
        }

    def _generate_gear_assembly_plan(self) -> dict:
        task_graph = [
            {"id": "t1", "action": "open_drawer", "parameters": ["parts_drawer"], "depends_on": []},
            {"id": "t2", "action": "pick_tray", "parameters": ["gear_tray"], "depends_on": ["t1"]},
            {"id": "t3", "action": "place_tray", "parameters": ["gear_tray", "parts_zone"], "depends_on": ["t2"]},
            {"id": "t4", "action": "pick_tool", "parameters": ["screwdriver"], "depends_on": ["t1"]},
            {"id": "t5", "action": "place_tool", "parameters": ["screwdriver", "tool_zone"], "depends_on": ["t4"]},
            {"id": "t6", "action": "hold_gearbase", "parameters": ["gearbase"], "depends_on": []},
            {"id": "t7", "action": "stabilize_gearbase", "parameters": ["gearbase"], "depends_on": ["t6"]},
            {"id": "t8", "action": "pick_gear", "parameters": ["gear"], "depends_on": ["t3"]},
            {"id": "t9", "action": "mount_gear", "parameters": ["gear", "shaft"], "depends_on": ["t7", "t8"]},
            {"id": "t10", "action": "pick_tool", "parameters": ["screwdriver"], "depends_on": ["t5"]},
            {"id": "t11", "action": "fasten_screw", "parameters": ["gearbase", "screwdriver"], "depends_on": ["t9", "t10"]},
            {"id": "t12", "action": "return_tool", "parameters": ["screwdriver", "parts_drawer"], "depends_on": ["t11"]},
            {"id": "t13", "action": "close_drawer", "parameters": ["parts_drawer"], "depends_on": ["t12"]},
        ]

        assignments = [
            {"task_id": "t1", "robot": "go2_z1"},
            {"task_id": "t2", "robot": "go2_z1"},
            {"task_id": "t3", "robot": "go2_z1"},
            {"task_id": "t4", "robot": "go2_z1"},
            {"task_id": "t5", "robot": "go2_z1"},
            {"task_id": "t6", "robot": "franka1"},
            {"task_id": "t7", "robot": "franka1"},
            {"task_id": "t8", "robot": "franka2"},
            {"task_id": "t9", "robot": "franka2"},
            {"task_id": "t10", "robot": "franka2"},
            {"task_id": "t11", "robot": "franka2"},
            {"task_id": "t12", "robot": "go2_z1"},
            {"task_id": "t13", "robot": "go2_z1"},
        ]

        synchronization = [
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

        behavior_trees = {
            "go2_z1": {
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
            },
            "franka1": {
                "type": "Sequence",
                "children": [
                    {"type": "Action", "name": "hold_gearbase", "parameters": ["gearbase"]},
                    {"type": "Action", "name": "stabilize_gearbase", "parameters": ["gearbase"]},
                ],
            },
            "franka2": {
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
            },
        }

        return {
            "task_id": self.task.get("task_id"),
            "instruction": self.task.get("instruction"),
            "task_graph": task_graph,
            "assignments": assignments,
            "synchronization": synchronization,
            "behavior_trees": behavior_trees,
        }
