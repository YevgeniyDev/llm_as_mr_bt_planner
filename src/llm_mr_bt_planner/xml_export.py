"""Serialize a canonical plan as BehaviorTree.CPP-style XML.

This module is an exporter only. It does not claim that ROS 2, BehaviorTree.CPP
plugins, robot skills, or hardware execution are installed.
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

from .bt import COMPOSITES, BTNode
from .plan import Plan


def export_behaviortree_cpp_xml(plan: Plan) -> str:
    """Render every robot tree from the canonical plan into one XML document."""
    lines = ['<?xml version="1.0"?>', '<root BTCPP_format="4">']
    for robot_id, tree in plan.behavior_trees.items():
        lines.append(f"  <BehaviorTree ID={quoteattr(robot_id)}>")
        lines.extend(_render_node(tree, indent=2))
        lines.append("  </BehaviorTree>")
    lines.append("</root>")
    return "\n".join(lines)


def _render_node(node: BTNode, indent: int) -> list[str]:
    pad = "  " * indent
    if node.type in COMPOSITES:
        tag = "Parallel" if node.type == "ParallelAll" else node.type
        composite_attributes = ""
        if node.type in {"Parallel", "ParallelAll"} and node.success_threshold is not None:
            composite_attributes = f" success_count={quoteattr(str(node.success_threshold))}"
        body = [line for child in node.children for line in _render_node(child, indent + 1)]
        return [f"{pad}<{tag}{composite_attributes}>", *body, f"{pad}</{tag}>"]

    tag = node.type if node.type in {
        "Action",
        "Condition",
        "WaitFor",
        "AcquireResource",
        "ReleaseResource",
    } else "InvalidNode"
    leaf_attributes = [
        f"name={quoteattr(node.name or '')}",
        f"params={quoteattr(';'.join(node.parameters))}",
    ]
    if node.node_id:
        leaf_attributes.append(f"node_id={quoteattr(node.node_id)}")
    if node.task_id:
        leaf_attributes.append(f"task_id={quoteattr(node.task_id)}")
    if node.type in {"WaitFor", "AcquireResource", "Action"} and node.timeout_ticks is not None:
        leaf_attributes.append(f"timeout_ticks={quoteattr(str(node.timeout_ticks))}")
    return [f"{pad}<{tag} {' '.join(leaf_attributes)}/>"]
