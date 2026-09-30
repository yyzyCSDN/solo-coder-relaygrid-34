import pytest

from relaygrid.storage import (
    SegmentRequest,
    StorageConfig,
    StorageScheduler,
    StorageState,
    degradation_cost_per_kwh,
    dispatch,
    plan_multi_period,
    soc,
    temperature_derate,
)


def make_config(**kw):
    defaults = dict(
        capacity_kwh=100.0,
        max_charge_kw=50.0,
        max_discharge_kw=50.0,
        min_soc=0.1,
        max_soc=0.9,
        charge_eff=0.95,
        discharge_eff=0.95,
        reserve_soc=0.1,
        degradation_cost_per_kwh=0.05,
    )
    defaults.update(kw)
    return StorageConfig(**defaults)


# --- legacy API still works -------------------------------------------------

def test_legacy_soc_and_dispatch():
    assert soc(StorageState(50, 100)) == 0.5
    kw, st = dispatch(StorageState(50, 100, 0.1, 0.9), 30, 1.0)
    assert kw == 30
    assert st.energy_kwh == pytest.approx(50 - 30 / 0.95)


# --- temperature derating ---------------------------------------------------

def test_temperature_derate_interpolation_and_clamp():
    cfg = make_config(temp_derate_curve=((-10, 1.0), (25, 1.0), (40, 0.6), (50, 0.2)))
    assert temperature_derate(cfg, 0) == 1.0
    assert temperature_derate(cfg, 25) == 1.0
    assert temperature_derate(cfg, 32.5) == pytest.approx(0.8)
    assert temperature_derate(cfg, 60) == 0.2
    assert temperature_derate(cfg, -20) == 1.0


def test_temperature_limits_power_and_is_reported():
    # 50 kW nameplate, derate 0.5 at 45 C -> 25 kW cap for a 1 h segment.
    cfg = make_config(
        temp_derate_curve=((40, 0.8), (45, 0.5), (50, 0.2)),
        reserve_soc=0.0,
    )
    rep = plan_multi_period(cfg, 50.0, [SegmentRequest(1.0, 40.0, ambient_c=45.0)])
    seg = rep.segments[0]
    assert seg.delivered_kw == pytest.approx(25.0)
    assert seg.temp_derate == 0.5
    assert seg.binding_constraint == "temperature"
    assert "temperature" in seg.reason


# --- efficiency + degradation ----------------------------------------------

def test_charge_discharge_efficiencies_and_degradation_cost():
    cfg = make_config(reserve_soc=0.0)
    # Charge 20 kW at terminal for 1 h -> battery gains 20*0.95 = 19 kWh.
    rep = plan_multi_period(cfg, 50.0, [SegmentRequest(1.0, -20.0, price_per_kwh=0.10)])
    seg = rep.segments[0]
    assert seg.delivered_kw == -20.0
    assert seg.energy_kwh == pytest.approx(-20.0)
    assert seg.battery_kwh == pytest.approx(19.0)
    assert rep.final_energy_kwh == pytest.approx(69.0)
    assert seg.energy_cost == pytest.approx(2.0)            # buying 20 kWh
    assert seg.degradation_cost == pytest.approx(19.0 * 0.05)
    assert seg.total_cost == pytest.approx(2.0 + 19.0 * 0.05)


def test_degradation_cost_helper():
    # $100/kWh replacement, 3000 cycles at 100 kWh usable -> 300000 kWh life
    assert degradation_cost_per_kwh(30000.0, 300000.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        degradation_cost_per_kwh(100.0, 0.0)


# --- SOC limits + reserve ---------------------------------------------------

def test_discharge_respects_reserve_floor():
    # 50 kWh stored; floor = 10% min + 10% reserve = 20 kWh.
    # Deliverable for 1 h at eff 0.95: (50-20)*0.95 = 28.5 kW.
    cfg = make_config(reserve_soc=0.1)
    rep = plan_multi_period(cfg, 50.0, [SegmentRequest(1.0, 50.0, price_per_kwh=1.0)])
    seg = rep.segments[0]
    assert seg.delivered_kw == pytest.approx(28.5)
    assert rep.final_soc == pytest.approx(0.2)
    assert seg.binding_constraint == "reserve"
    # Revenue is negative cost.
    assert seg.energy_cost == pytest.approx(-28.5)


def test_discharge_without_reserve_hits_soc_lower():
    cfg = make_config(reserve_soc=0.0)
    rep = plan_multi_period(cfg, 15.0, [SegmentRequest(1.0, 50.0)])
    seg = rep.segments[0]
    # floor 10 kWh; (15-10)*0.95 = 4.75 kW
    assert seg.delivered_kw == pytest.approx(4.75)
    assert rep.final_soc == pytest.approx(0.1)
    assert seg.binding_constraint == "soc_lower"


def test_charge_respects_soc_upper_limit():
    cfg = make_config(reserve_soc=0.0)
    # 85 kWh stored, max 90; absorbable = (90-85)/(0.95*1) = 5.263 kW
    rep = plan_multi_period(cfg, 85.0, [SegmentRequest(1.0, -50.0)])
    seg = rep.segments[0]
    assert seg.delivered_kw == pytest.approx(-5.2631578947)
    assert rep.final_soc == pytest.approx(0.9)
    assert seg.binding_constraint == "soc_upper"


def test_power_rating_binds():
    cfg = make_config(reserve_soc=0.0)
    rep = plan_multi_period(cfg, 85.0, [SegmentRequest(1.0, 50.0)])
    # Energy headroom is ample; 50 kW rating is exactly met -> "none"
    assert rep.segments[0].binding_constraint == "none"
    cfg2 = make_config(reserve_soc=0.0, max_discharge_kw=30.0)
    rep2 = plan_multi_period(cfg2, 85.0, [SegmentRequest(1.0, 50.0)])
    assert rep2.segments[0].delivered_kw == 30.0
    assert rep2.segments[0].binding_constraint == "power"


def test_multi_period_soc_stays_in_band():
    cfg = make_config(reserve_soc=0.05)
    segs = [
        SegmentRequest(1.0, 40.0, price_per_kwh=1.0),   # discharge
        SegmentRequest(1.0, -40.0, price_per_kwh=0.1),  # charge back
        SegmentRequest(1.0, 40.0, price_per_kwh=1.0),   # discharge again
    ]
    rep = plan_multi_period(cfg, 60.0, segs)
    e_min = 100 * (0.1 + 0.05)
    e_max = 100 * 0.9
    for s in rep.segments:
        assert e_min - 1e-9 <= s.soc_end * 100 <= e_max + 1e-9
    assert rep.final_energy_kwh < 60.0  # round-trip losses
    assert rep.total_cost == pytest.approx(sum(s.total_cost for s in rep.segments))


def test_reserve_does_not_fit_between_limits_rejected():
    with pytest.raises(ValueError):
        make_config(min_soc=0.5, max_soc=0.6, reserve_soc=0.2)


def test_initial_soc_outside_band_rejected():
    cfg = make_config()
    with pytest.raises(ValueError):
        plan_multi_period(cfg, 5.0, [SegmentRequest(1.0, 10.0)])
    with pytest.raises(ValueError):
        plan_multi_period(cfg, 95.0, [SegmentRequest(1.0, 10.0)])


# --- atomicity: failure and cancellation ------------------------------------

def test_strict_failure_rolls_back_entire_plan():
    cfg = make_config(reserve_soc=0.1)
    sched = StorageScheduler(cfg, 25.0)  # only 5 kWh above reserve floor
    segs = [
        SegmentRequest(1.0, 4.75, price_per_kwh=1.0, label="ok"),
        SegmentRequest(1.0, 40.0, price_per_kwh=1.0, label="too-big"),
    ]
    rep = sched.commit(segs, strict=True)
    assert rep.committed is False
    assert rep.failure is not None and "too-big" in rep.failure
    assert sched.energy_kwh == 25.0          # no partial energy change
    assert rep.final_energy_kwh == 25.0
    assert rep.total_cost == 0.0


def test_non_strict_plan_partially_dispatches_and_commits():
    cfg = make_config(reserve_soc=0.1)
    sched = StorageScheduler(cfg, 25.0)
    rep = sched.commit([SegmentRequest(1.0, 40.0, label="big")])
    assert rep.committed is True
    assert rep.segments[0].binding_constraint == "reserve"
    assert sched.energy_kwh == pytest.approx(20.0)  # drained to reserve floor


def test_cancel_leaves_state_untouched_and_zeroes_costs():
    cfg = make_config()
    sched = StorageScheduler(cfg, 50.0)
    planned = [SegmentRequest(1.0, 20.0), SegmentRequest(1.0, -20.0)]
    rep = sched.cancel(planned, reason="price forecast withdrawn")
    assert rep.committed is False
    assert "withdrawn" in rep.failure
    assert all(s.delivered_kw == 0.0 for s in rep.segments)
    assert all(s.binding_constraint == "cancelled" for s in rep.segments)
    assert all(s.total_cost == 0.0 for s in rep.segments)
    assert sched.energy_kwh == 50.0
    # Preview never mutates either.
    before = sched.energy_kwh
    sched.preview([SegmentRequest(1.0, 30.0)])
    assert sched.energy_kwh == before


def test_segment_report_fields_and_reasons():
    cfg = make_config(
        temp_derate_curve=((40, 0.5),),
    )
    rep = plan_multi_period(
        cfg, 50.0,
        [SegmentRequest(0.5, -100.0, price_per_kwh=0.2, ambient_c=42.0, label="hot-charge")],
    )
    s = rep.segments[0]
    assert s.label == "hot-charge"
    assert s.duration_h == 0.5
    # 50 kW nameplate * 0.5 derate = 25 kW charge head; SOC headroom is larger.
    assert s.delivered_kw == pytest.approx(-25.0)
    assert s.binding_constraint == "temperature"
    assert s.energy_cost == pytest.approx(12.5 * 0.2)
    assert s.battery_kwh == pytest.approx(25.0 * 0.5 * 0.95)
