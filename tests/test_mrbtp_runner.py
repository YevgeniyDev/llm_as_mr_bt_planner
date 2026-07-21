from scripts.run_mrbtp import _grounded_conditions


class _Agent:
    def __init__(self, conditions):
        self.expanded_condition_dict = {frozenset(condition): object() for condition in conditions}


class _Planner:
    def __init__(self, conditions):
        self.planned_agent_list = [_Agent(conditions)]


def test_grounded_condition_requires_initial_state_support():
    planner = _Planner([{"goal"}, {"reachable"}, {"missing"}])
    grounded = _grounded_conditions(planner, frozenset({"reachable", "initial"}))
    assert grounded == [frozenset({"reachable"})]


def test_frontier_exhaustion_has_no_grounded_condition():
    planner = _Planner([{"goal"}, {"missing"}])
    assert _grounded_conditions(planner, frozenset({"initial"})) == []
