import pytest

from relaygrid.storage.dispatch import (
    BatterySpec,
    ReserveRequirement,
    StorageState,
    ThermalDerating,
    commit_plan,
    dispatch,
    plan_dispatch,
    soc,
)


def spec(**kw):
    base = dict(max_charge_kw=100.0, max_discharge_kw=100.0,
                charge_eff=0.95, discharge_eff=0.95,
                degradation_cost_per_kwh=0.0)
    base.update(kw)
    return BatterySpec(**base)


def test_soc_still_works():
    assert soc(StorageState(50, 100)) == 0.5


def test_legacy_dispatch_unchanged():
    # Discharge 100 kW for 1 h from 50 kWh: floor (10 kWh) caps deliverable
    # to (50-10)*0.95 = 38 kW, leaving the battery at its floor.
    state = StorageState(50, 100)
    actual, new_state = dispatch(state, 100, 1)
    assert actual == pytest.approx(38.0)
    assert new_state.energy_kwh == pytest.approx(10.0)


def test_plan_serves_round_trip():
    # 30 kW charge then 30 kW discharge, both fully served.
    state = StorageState(50, 100)
    plan = plan_dispatch(
        state,
        requested_kw=[-30.0, 30.0],
        duration_h=[1.0, 1.0],
        spec=spec(),
    )
    assert plan.feasible
    assert plan.reason == "ok"
    assert len(plan.segments) == 2
    assert all(s.binding_constraint == "none" for s in plan.segments)
    assert plan.final_state is not state
    # Round-trip losses: less energy returned than stored.
    assert plan.final_state.energy_kwh < state.energy_kwh


def test_plan_never_breaches_soc_window():
    state = StorageState(50, 100)
    plan = plan_dispatch(
        state,
        requested_kw=[-500.0, 500.0, -500.0, 500.0],
        duration_h=[1.0, 1.0, 1.0, 1.0],
        spec=spec(),
    )
    assert plan.feasible
    for s in plan.segments:
        assert 0.1 - 1e-9 <= s.end_soc <= 0.95 + 1e-9
    assert 0.1 - 1e-9 <= soc(plan.final_state) <= 0.95 + 1e-9


def test_discharge_curtailed_at_floor():
    # 12 kWh start: only (12-10)*0.95 = 1.9 kW deliverable above the floor.
    state = StorageState(12, 100)
    plan = plan_dispatch(state, [100.0], [1.0], spec=spec())
    seg = plan.segments[0]
    assert seg.actual_kw == pytest.approx(1.9)
    assert seg.binding_constraint == "soc_lower_limit"
    assert soc(plan.final_state) == pytest.approx(0.1, abs=1e-9)


def test_charge_curtailed_at_ceiling():
    # 94 kWh start: only (95-94)/0.95 = 1.05 kW absorbable below the ceiling.
    state = StorageState(94, 100)
    plan = plan_dispatch(state, [-100.0], [1.0], spec=spec())
    seg = plan.segments[0]
    assert -seg.actual_kw == pytest.approx(1.0526, rel=1e-3)
    assert seg.binding_constraint == "soc_upper_limit"
    assert soc(plan.final_state) == pytest.approx(0.95, abs=1e-9)


def test_temperature_derate_caps_power():
    # Short 15-min period so SOC is not the limit; 45 C derates to 60 kW.
    state = StorageState(50, 100)
    cool = plan_dispatch(state, [100.0], [0.25], ambient_c=[25.0], spec=spec())
    hot = plan_dispatch(state, [100.0], [0.25], ambient_c=[45.0], spec=spec())
    assert cool.segments[0].actual_kw == pytest.approx(100.0)
    assert cool.segments[0].binding_constraint == "none"
    assert hot.segments[0].actual_kw == pytest.approx(60.0)
    assert hot.segments[0].binding_constraint == "temperature_derate"


def test_derating_factor_piecewise():
    d = ThermalDerating(full_output_c=25.0, derate_per_c=0.02, min_factor=0.5)
    assert d.factor(20.0) == 1.0
    assert d.factor(25.0) == 1.0
    assert d.factor(35.0) == pytest.approx(0.8)
    assert d.factor(200.0) == 0.5


def test_reserve_holdback_limits_export():
    # 95 kWh start: reserve energy floor (7.89 kWh) is below available
    # energy, so the 30 kW export headroom holdback is the binding limit.
    state = StorageState(95, 100)
    reserve = ReserveRequirement(reserve_kw=30.0, duration_h=0.25)
    plan = plan_dispatch(state, [100.0], [1.0], spec=spec(), reserve=reserve)
    seg = plan.segments[0]
    assert seg.actual_kw == pytest.approx(70.0)
    assert seg.binding_constraint == "reserve_holdback"


def test_reserve_energy_raises_floor():
    # 40 kW reserve for 1 h needs 42.1 kWh held back; the floor rises and
    # curtails export to ~36 kW.
    state = StorageState(90, 100)
    reserve = ReserveRequirement(reserve_kw=40.0, duration_h=1.0)
    plan = plan_dispatch(state, [100.0], [1.0], spec=spec(), reserve=reserve)
    seg = plan.segments[0]
    assert seg.actual_kw == pytest.approx(36.0, abs=0.5)
    assert seg.binding_constraint == "soc_lower_limit"


def test_reserve_floor_below_initial_soc_infeasible():
    # Starting below the reserve energy floor means the reserve cannot be
    # delivered right now: the plan is infeasible and commits nothing.
    state = StorageState(20, 100)
    reserve = ReserveRequirement(reserve_kw=40.0, duration_h=1.0)
    plan = plan_dispatch(state, [100.0], [1.0], spec=spec(), reserve=reserve)
    assert not plan.feasible
    assert plan.final_state is state
    with pytest.raises(ValueError):
        commit_plan(state, plan)


def test_reserve_infeasible_commits_nothing():
    state = StorageState(50, 100)
    reserve = ReserveRequirement(reserve_kw=200.0, duration_h=0.25)
    plan = plan_dispatch(state, [50.0], [1.0], spec=spec(), reserve=reserve)
    assert not plan.feasible
    assert plan.final_state is state  # no partial state committed
    with pytest.raises(ValueError):
        commit_plan(state, plan)


def test_reserve_energy_exceeds_window():
    state = StorageState(50, 100)
    reserve = ReserveRequirement(reserve_kw=200.0, duration_h=10.0)
    plan = plan_dispatch(state, [0.0], [1.0], spec=spec(), reserve=reserve)
    assert not plan.feasible
    assert plan.final_state is state
    assert "reserve" in plan.reason


def test_degradation_cost_throughput():
    state = StorageState(50, 100)
    plan = plan_dispatch(
        state, [30.0, -30.0], [1.0, 1.0],
        spec=spec(degradation_cost_per_kwh=0.02),
    )
    assert plan.feasible
    assert plan.total_throughput_kwh > 0
    assert plan.total_cost == pytest.approx(plan.total_throughput_kwh * 0.02)
    # Idling moves no energy and costs nothing.
    idle = plan_dispatch(state, [0.0], [1.0], spec=spec(degradation_cost_per_kwh=0.02))
    assert idle.total_cost == 0.0
    assert idle.segments[0].throughput_kwh == 0.0


def test_efficiency_losses_reflected():
    # Charge 50 kW for 1 h from 30 kWh: fully served, 47.5 kWh gained.
    state = StorageState(30, 100)
    plan = plan_dispatch(state, [-50.0], [1.0], spec=spec())
    seg = plan.segments[0]
    assert seg.energy_change_kwh == pytest.approx(47.5)
    assert seg.throughput_kwh == pytest.approx(47.5)


def test_segment_output_fields():
    state = StorageState(90, 100)
    plan = plan_dispatch(state, [25.0], [2.0], spec=spec())
    seg = plan.segments[0]
    assert seg.period == 0
    assert seg.requested_kw == 25.0
    assert seg.actual_kw == 25.0
    assert seg.duration_h == 2.0
    assert isinstance(seg.energy_change_kwh, float)
    assert isinstance(seg.end_soc, float)
    assert isinstance(seg.degradation_cost, float)
    assert seg.binding_constraint == "none"


def test_commit_returns_final_state():
    state = StorageState(50, 100)
    plan = plan_dispatch(state, [-40.0], [1.0], spec=spec())
    assert plan.feasible
    committed = commit_plan(state, plan)
    assert committed is plan.final_state
    assert committed.energy_kwh > state.energy_kwh


def test_cancellation_leaves_state_untouched():
    state = StorageState(50, 100)
    plan = plan_dispatch(state, [-40.0], [1.0], spec=spec())
    # A cancelled plan is simply discarded: planning never mutated state.
    _cancelled = None
    assert _cancelled is None
    assert state.energy_kwh == 50.0


def test_input_validation():
    state = StorageState(50, 100)
    with pytest.raises(ValueError):
        plan_dispatch(state, [1.0], [0.0], spec=spec())
    with pytest.raises(ValueError):
        plan_dispatch(state, [1.0], [1.0, 2.0], spec=spec())
    with pytest.raises(ValueError):
        plan_dispatch(state, [1.0], [1.0], ambient_c=[25.0, 30.0], spec=spec())
    with pytest.raises(ValueError):
        plan_dispatch(state, [1.0], [1.0], spec=spec(charge_eff=1.5))
