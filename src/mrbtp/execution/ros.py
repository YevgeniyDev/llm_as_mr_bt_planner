"""ROS / real-robot execution backend (scaffold).

This is the seam where simulation-first work meets real hardware. It does not
control robots yet - instead it provides:

* :func:`export_behaviortree_cpp_xml`, which serializes a generated tree to the
  BehaviorTree.CPP XML dialect used by Nav2 / py_trees_ros, so the exact tree
  the simulator validated can be loaded by a real executor; and
* :class:`RosExecutionBackend`, which raises a clear, actionable
  :class:`NotImplementedError` describing what an integrator must wire up
  (action/skill servers per leaf, a blackboard for predicates, condition
  monitors from perception).

Keeping the contract here means the planner and experiment layers already speak
to the real-robot backend through the same :class:`ExecutionBackend` interface.
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

from ..bt import BTNode
from ..domain import Scenario
from ..plan import Plan
from .base import ExecutionResult


def export_behaviortree_cpp_xml(plan: Plan, tree_id: str = "MultiRobotPlan") -> str:
    """Render the plan's per-robot trees as BehaviorTree.CPP XML.

    Each robot becomes a top-level ``BehaviorTree`` whose id is the robot id.
    Action and Condition leaves map to ``<Action>`` / ``<Condition>`` nodes whose
    ``name`` is the capability/predicate and whose parameters are passed as a
    single ``params`` port (a list the skill server can parse). Composites map
    1:1 to ``<Sequence>`` / ``<Fallback>`` / ``<Parallel>``.
    """
    lines = ['<?xml version="1.0"?>', '<root BTCPP_format="4">']
    for robot_id, tree in plan.behavior_trees.items():
        lines.append(f"  <BehaviorTree ID={quoteattr(robot_id)}>")
        lines.extend(_render_node(tree, indent=2))
        lines.append("  </BehaviorTree>")
    lines.append("</root>")
    return "\n".join(lines)


def _render_node(node: BTNode, indent: int) -> list[str]:
    pad = "  " * indent
    if node.type in {"Sequence", "Fallback", "Parallel"}:
        attrs = ""
        if node.type == "Parallel" and node.success_threshold is not None:
            attrs = f" success_count={quoteattr(str(node.success_threshold))}"
        opened = f"{pad}<{node.type}{attrs}>"
        body: list[str] = []
        for child in node.children:
            body.extend(_render_node(child, indent + 1))
        return [opened, *body, f"{pad}</{node.type}>"]
    params = ";".join(node.parameters)
    return [f"{pad}<{node.type} name={quoteattr(node.name or '')} params={quoteattr(params)}/>"]


class RosExecutionBackend:
    name = "ros"

    def __init__(self, **kwargs) -> None:
        self.config = kwargs

    def execute(self, plan: Plan, scenario: Scenario) -> ExecutionResult:
        raise NotImplementedError(
            "RosExecutionBackend is a scaffold for real-robot testing. To enable it:\n"
            "  1. Stand up an action/skill server per capability (the leaf 'name' selects it; "
            "'parameters' are the goal).\n"
            "  2. Back the symbolic predicates with a blackboard updated by perception/condition monitors.\n"
            "  3. Load the tree via export_behaviortree_cpp_xml(plan) into py_trees_ros / BehaviorTree.CPP.\n"
            "Until then, use SymbolicExecutionBackend."
        )
