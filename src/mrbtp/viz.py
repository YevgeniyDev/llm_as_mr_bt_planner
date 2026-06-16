"""Behavior Tree visualization.

Renders the per-robot trees of a :class:`mrbtp.plan.Plan` as Mermaid
``flowchart`` diagrams and bundles them into a single self-contained HTML report.
The report pulls Mermaid.js from a CDN (so the file has no local dependencies and
opens directly in a browser); the diagram *definitions* are plain text, so they
also paste straight into GitHub Markdown or https://mermaid.live.

Node shapes: composites (Sequence/Fallback/Parallel) are rectangles, Actions are
stadiums, Conditions are hexagons.
"""

from __future__ import annotations

import html
from typing import Any

from .bt import BTNode
from .plan import Plan

_MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"

_CLASSDEFS = [
    "classDef composite fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;",
    "classDef action fill:#dcfce7,stroke:#166534,color:#14532d;",
    "classDef condition fill:#fef9c3,stroke:#854d0e,color:#713f12;",
    "classDef other fill:#fee2e2,stroke:#991b1b,color:#7f1d1d;",
]


def _node_label(node: BTNode) -> str:
    if node.is_leaf:
        if node.parameters:
            return f"{node.name}({', '.join(node.parameters)})"
        return str(node.name)
    if node.type == "Parallel" and node.success_threshold is not None:
        return f"Parallel ({node.success_threshold})"
    return node.type


def _category(node: BTNode) -> str:
    if node.type == "Action":
        return "action"
    if node.type == "Condition":
        return "condition"
    if node.type in {"Sequence", "Fallback", "Parallel"}:
        return "composite"
    return "other"


def _shape(node_id: str, label: str, category: str) -> str:
    # Mermaid wraps quoted labels; escape any embedded quotes as HTML entities.
    text = label.replace('"', "&quot;")
    if category == "action":
        return f'{node_id}(["{text}"])'
    if category == "condition":
        return f'{node_id}{{{{"{text}"}}}}'
    return f'{node_id}["{text}"]'


def bt_to_mermaid(tree: BTNode, direction: str = "TD") -> str:
    """Return a Mermaid ``flowchart`` definition for one behavior tree."""
    lines = [f"flowchart {direction}"]
    by_category: dict[str, list[str]] = {"composite": [], "action": [], "condition": [], "other": []}
    counter = {"n": 0}

    def fresh() -> str:
        node_id = f"n{counter['n']}"
        counter["n"] += 1
        return node_id

    def walk(node: BTNode) -> str:
        node_id = fresh()
        category = _category(node)
        by_category[category].append(node_id)
        lines.append(f"    {_shape(node_id, _node_label(node), category)}")
        for child in node.children:
            child_id = walk(child)
            lines.append(f"    {node_id} --> {child_id}")
        return node_id

    walk(tree)
    lines.extend(f"    {definition}" for definition in _CLASSDEFS)
    for category, ids in by_category.items():
        if ids:
            lines.append(f"    class {','.join(ids)} {category};")
    return "\n".join(lines)


def plan_to_html(
    plan: Plan,
    title: str = "Behavior Trees",
    meta: dict[str, Any] | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> str:
    """Render a self-contained HTML report.

    Always includes a "Behavior Trees" tab (one Mermaid diagram per robot). When
    ``trace`` (a simulation execution trace) is provided, also adds an
    "Action Plan" tab: a chronological table of every robot's BT node in the
    order it fired.
    """
    meta = meta or {}
    meta_html = "".join(
        f'<span class="pill"><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</span>'
        for k, v in meta.items()
    )
    trees_html = _trees_section(plan)

    tabs = [("trees", "Behavior Trees", trees_html)]
    if trace is not None:
        tabs.append(("plan", "Action Plan", _action_plan_section(trace)))

    nav_parts = []
    section_parts = []
    for index, (tab_id, label, body) in enumerate(tabs):
        button_class = ' class="active"' if index == 0 else ""
        tab_class = "tab active" if index == 0 else "tab"
        nav_parts.append(f'<button data-tab="{tab_id}"{button_class}>{html.escape(label)}</button>')
        section_parts.append(f'<div id="{tab_id}" class="{tab_class}">{body}</div>')
    nav = "".join(nav_parts)
    sections = "".join(section_parts)

    return _TEMPLATE.format(
        title=html.escape(title),
        meta=meta_html,
        nav=nav,
        sections=sections,
        cdn=_MERMAID_CDN,
    )


def _trees_section(plan: Plan) -> str:
    diagrams = [_LEGEND]
    for robot_id, tree in plan.behavior_trees.items():
        diagram = bt_to_mermaid(tree)
        diagrams.append(
            f'<section><h2>{html.escape(robot_id)}</h2>'
            f'<pre class="mermaid">\n{html.escape(diagram)}\n</pre></section>'
        )
    if len(diagrams) == 1:
        diagrams.append("<p><em>No behavior trees in this plan.</em></p>")
    return "\n".join(diagrams)


def _action_plan_section(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return "<p><em>No execution trace (the plan was invalid or not simulated).</em></p>"

    robots = list(dict.fromkeys(event.get("robot", "") for event in trace))
    robot_class = {robot: f"rc{index % 6}" for index, robot in enumerate(robots)}

    rows = []
    for step, event in enumerate(trace, start=1):
        robot = str(event.get("robot", ""))
        tick = str(event.get("tick", ""))
        if event.get("event") == "action":
            node_type, node = "Action", str(event.get("action", ""))
            detail = _format_effects(event.get("effects", {}))
        else:
            node_type, node = "Condition", str(event.get("condition", ""))
            detail = "wait satisfied"
        rows.append(
            f"<tr>"
            f"<td class='num'>{step}</td>"
            f"<td class='num'>{html.escape(tick)}</td>"
            f"<td><span class='robot {robot_class.get(robot, '')}'>{html.escape(robot)}</span></td>"
            f"<td>{node_type}</td>"
            f"<td><code>{html.escape(node)}</code></td>"
            f"<td class='eff'>{detail}</td>"
            f"</tr>"
        )

    legend = "".join(
        f'<span class="robot {robot_class[robot]}">{html.escape(robot)}</span>' for robot in robots
    )
    return (
        f'<div class="legend">{legend}</div>'
        "<table class='plan'>"
        "<thead><tr><th>#</th><th>tick</th><th>robot</th><th>type</th><th>node</th>"
        "<th>effects / sync</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _format_effects(effects: dict[str, Any]) -> str:
    parts = [f'<span class="add">+{html.escape(a)}</span>' for a in effects.get("add", [])]
    parts += [f'<span class="del">-{html.escape(d)}</span>' for d in effects.get("delete", [])]
    return " ".join(parts) if parts else "—"


_LEGEND = (
    '<div class="legend">'
    '<span class="pill comp">Sequence / Fallback / Parallel</span>'
    '<span class="pill act">Action</span>'
    '<span class="pill cond">Condition</span>'
    "</div>"
)

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
  h1 {{ margin-bottom: .25rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: .25rem; }}
  .meta, .legend {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0 1rem; }}
  .pill {{ background: #f3f4f6; border-radius: 999px; padding: .15rem .6rem; font-size: .85rem; }}
  .pill.comp {{ background: #dbeafe; }} .pill.act {{ background: #dcfce7; }} .pill.cond {{ background: #fef9c3; }}
  section {{ overflow-x: auto; }}
  pre.mermaid {{ background: #fff; }}
  nav.tabs {{ display: flex; gap: .25rem; margin: 1rem 0; border-bottom: 2px solid #e5e7eb; }}
  nav.tabs button {{ border: none; background: none; padding: .5rem 1rem; font-size: 1rem;
                     cursor: pointer; color: #6b7280; border-bottom: 2px solid transparent; margin-bottom: -2px; }}
  nav.tabs button.active {{ color: #1e3a8a; border-bottom-color: #1e40af; font-weight: 600; }}
  .tab {{ display: none; }} .tab.active {{ display: block; }}
  table.plan {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  table.plan th, table.plan td {{ border: 1px solid #e5e7eb; padding: .35rem .6rem; text-align: left; }}
  table.plan th {{ background: #f9fafb; }}
  table.plan td.num {{ text-align: right; color: #6b7280; }}
  table.plan code {{ background: #f3f4f6; padding: .05rem .3rem; border-radius: 4px; }}
  .eff .add {{ color: #166534; }} .eff .del {{ color: #991b1b; }}
  .robot {{ border-radius: 999px; padding: .1rem .5rem; font-size: .8rem; }}
  .rc0 {{ background: #dbeafe; }} .rc1 {{ background: #dcfce7; }} .rc2 {{ background: #fef9c3; }}
  .rc3 {{ background: #fce7f3; }} .rc4 {{ background: #e0e7ff; }} .rc5 {{ background: #ffedd5; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{meta}</div>
<nav class="tabs">{nav}</nav>
{sections}
<script type="module">
  import mermaid from "{cdn}";
  mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
</script>
<script>
  document.querySelectorAll('nav.tabs button').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      document.querySelectorAll('nav.tabs button').forEach(function (b) {{ b.classList.remove('active'); }});
      document.querySelectorAll('.tab').forEach(function (t) {{ t.classList.remove('active'); }});
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    }});
  }});
</script>
</body>
</html>
"""
