from __future__ import annotations

from llm_mr_bt_planner.bt import BTParseError, iter_leaves, iter_nodes, parse_node
from llm_mr_bt_planner.execution import export_behaviortree_cpp_xml


def test_parse_nested_tree_and_iter_leaves():
    tree = parse_node(
        {
            "type": "Sequence",
            "children": [
                {"type": "Condition", "name": "ready", "parameters": ["x"]},
                {
                    "type": "Fallback",
                    "children": [
                        {"type": "Action", "name": "a", "parameters": []},
                        {"type": "Action", "name": "b", "parameters": ["y"]},
                    ],
                },
            ],
        }
    )
    leaves = list(iter_leaves(tree))
    assert [leaf.name for leaf in leaves] == ["ready", "a", "b"]
    assert len(list(iter_nodes(tree))) == 5


def test_to_dict_roundtrip():
    data = {
        "type": "Sequence",
        "children": [{"type": "Action", "name": "a", "parameters": ["x"]}],
    }
    assert parse_node(data).to_dict() == data


def test_parse_node_rejects_non_object():
    try:
        parse_node(["not", "an", "object"])
    except BTParseError:
        return
    raise AssertionError("expected BTParseError")


def test_export_behaviortree_cpp_xml(toy_plan):
    xml = export_behaviortree_cpp_xml(toy_plan)
    assert '<BehaviorTree ID="A">' in xml
    assert '<Action name="make"' in xml
    assert '<Condition name="p"' in xml
    assert xml.count("<BehaviorTree") == 2
