from __future__ import annotations

from mrbtp.simulation import simulate
from mrbtp.viz import bt_to_mermaid, plan_to_html


def test_bt_to_mermaid_shapes(toy_plan):
    diagram = bt_to_mermaid(toy_plan.behavior_trees["B"])
    assert diagram.startswith("flowchart TD")
    assert '(["use"])' in diagram          # Action -> stadium
    assert '{{"p"}}' in diagram             # Condition -> hexagon
    assert "classDef action" in diagram
    assert "-->" in diagram


def test_mermaid_has_one_edge_per_child(toy_plan):
    # B: Sequence + Condition + Action = 3 nodes, 2 edges.
    assert bt_to_mermaid(toy_plan.behavior_trees["B"]).count("-->") == 2


def test_plan_to_html_bundles_every_robot(toy_plan):
    out = plan_to_html(toy_plan, title="T", meta={"valid": True})
    assert "<!doctype html>" in out.lower()
    assert "mermaid" in out
    for robot in toy_plan.behavior_trees:
        assert f"<h2>{robot}</h2>" in out
    assert out.count('<pre class="mermaid">') == len(toy_plan.behavior_trees)


def test_action_plan_tab_renders_trace(toy_scenario, toy_plan):
    report = simulate(toy_plan, toy_scenario)
    out = plan_to_html(toy_plan, trace=report.trace)
    assert '<button data-tab="plan"' in out          # the Action Plan tab exists
    assert "<table class='plan'>" in out
    actions = [e for e in report.trace if e["event"] == "action"]
    assert out.count("<tr>") >= len(actions)
    assert "make" in out and "use" in out


def test_no_trace_means_no_action_plan_tab(toy_plan):
    out = plan_to_html(toy_plan)  # trace omitted
    assert '<button data-tab="plan"' not in out
    assert '<button data-tab="trees"' in out
