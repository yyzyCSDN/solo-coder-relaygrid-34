from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

EPS = 1e-9


@dataclass(frozen=True)
class StorageState:
    energy_kwh: float
    capacity_kwh: float
    min_soc: float = 0.1
    max_soc: float = 0.95


@dataclass(frozen=True)
class ThermalDerating:
    """Temperature-dependent derating of battery power capability.

    Available power is full nameplate up to ``full_output_c`` and then
    falls linearly with ambient temperature down to ``min_factor``.
    """

    full_output_c: float = 25.0
    derate_per_c: float = 0.02
    min_factor: float = 0.5

    def factor(self, ambient_c: float) -> float:
        if self.derate_per_c < 0.0 or not 0.0 <= self.min_factor <= 1.0:
            raise ValueError("invalid thermal derating parameters")
        raw = 1.0 - self.derate_per_c * max(0.0, ambient_c - self.full_output_c)
        return min(1.0, max(self.min_factor, raw))


@dataclass(frozen=True)
class ReserveRequirement:
    """Mandatory contingency reserve that must stay available.

    The battery must be able to deliver ``reserve_kw`` for ``duration_h``
    on top of any scheduled discharge, so both export headroom and the
    usable SOC window are held back.
    """

    reserve_kw: float = 0.0
    duration_h: float = 0.25


@dataclass(frozen=True)
class BatterySpec:
    """Nameplate power, round-trip efficiency and degradation economics."""

    max_charge_kw: float
    max_discharge_kw: float
    charge_eff: float = 0.95
    discharge_eff: float = 0.95
    degradation_cost_per_kwh: float = 0.0


@dataclass(frozen=True)
class PlanSegment:
    """One period of a multi-period storage dispatch plan."""

    period: int
    requested_kw: float        # grid-side signed power: + discharge, - charge
    actual_kw: float           # grid-side power actually committed
    duration_h: float
    energy_change_kwh: float   # battery DC energy change: + charge gain, - discharge drain
    end_soc: float
    throughput_kwh: float      # DC energy processed this period
    degradation_cost: float
    binding_constraint: str    # what capped the request, or "none"


@dataclass(frozen=True)
class DispatchPlan:
    """Result of multi-period storage planning.

    Planning is atomic: the input state is never mutated and ``final_state``
    is set only when the plan is feasible. An infeasible or cancelled plan
    commits nothing -- apply a feasible plan with :func:`commit_plan`.
    """

    feasible: bool
    segments: tuple[PlanSegment, ...]
    final_state: StorageState | None
    total_throughput_kwh: float
    total_cost: float
    reason: str


def dispatch(state: StorageState, requested_kw: float, duration_h: float, charge_eff: float = 0.95, discharge_eff: float = 0.95):
    if state.capacity_kwh <= 0 or duration_h < 0: raise ValueError("invalid storage state")
    if not 0 < charge_eff <= 1 or not 0 < discharge_eff <= 1: raise ValueError("invalid efficiency")
    min_e = state.capacity_kwh * state.min_soc; max_e = state.capacity_kwh * state.max_soc
    e = min(max(state.energy_kwh, min_e), max_e)
    if requested_kw >= 0:
        max_deliverable = max(0.0, (e - min_e) * discharge_eff / duration_h) if duration_h else 0.0
        actual = min(requested_kw, max_deliverable)
        new_e = e - actual * duration_h / discharge_eff
    else:
        charge_kw = -requested_kw
        max_absorb = max(0.0, (max_e - e) / (charge_eff * duration_h)) if duration_h else 0.0
        actual = -min(charge_kw, max_absorb)
        new_e = e + (-actual) * duration_h * charge_eff
    return actual, StorageState(new_e, state.capacity_kwh, state.min_soc, state.max_soc)


def soc(state: StorageState) -> float:
    return state.energy_kwh / state.capacity_kwh


def _default_spec() -> BatterySpec:
    # Effectively unlimited nameplate so SOC/temperature/reserve bind first.
    return BatterySpec(max_charge_kw=1e12, max_discharge_kw=1e12)


def plan_dispatch(
    state: StorageState,
    requested_kw: Sequence[float],
    duration_h: Sequence[float],
    ambient_c: Sequence[float] | None = None,
    spec: BatterySpec | None = None,
    reserve: ReserveRequirement | None = None,
    derating: ThermalDerating | None = None,
) -> DispatchPlan:
    """Build a multi-period dispatch plan without mutating ``state``.

    Each period is capped by the derated power capability, the contingency
    reserve that must stay available, and the SOC window (including the
    reserve energy that has to remain stored). Every segment reports the
    committed grid-side power, battery DC energy change, cycle throughput
    degradation cost and the constraint that bound the request.

    The returned plan is atomic: ``final_state`` is the state after the
    last period only when every period is feasible; otherwise it is the
    input state and nothing is committed.
    """

    if state.capacity_kwh <= 0.0:
        raise ValueError("invalid storage state")
    if not 0.0 <= state.min_soc < state.max_soc <= 1.0:
        raise ValueError("invalid SOC window")

    requests = list(requested_kw)
    durations = list(duration_h)
    n = len(requests)
    if n == 0:
        raise ValueError("dispatch plan must cover at least one period")
    if len(durations) != n:
        raise ValueError("requested_kw and duration_h length mismatch")
    if any(dt <= 0.0 for dt in durations):
        raise ValueError("each duration must be positive")

    if spec is None:
        spec = _default_spec()
    if spec.max_charge_kw < 0.0 or spec.max_discharge_kw < 0.0:
        raise ValueError("invalid battery spec")
    if not 0.0 < spec.charge_eff <= 1.0 or not 0.0 < spec.discharge_eff <= 1.0:
        raise ValueError("invalid efficiency")
    if spec.degradation_cost_per_kwh < 0.0:
        raise ValueError("invalid degradation cost")

    if reserve is None:
        reserve = ReserveRequirement()
    if reserve.reserve_kw < 0.0 or reserve.duration_h <= 0.0:
        raise ValueError("invalid reserve requirement")

    if derating is None:
        derating = ThermalDerating()
    if ambient_c is None:
        ambient = [derating.full_output_c] * n
    else:
        ambient = list(ambient_c)
        if len(ambient) != n:
            raise ValueError("ambient_c length mismatch")

    capacity = state.capacity_kwh
    min_e = capacity * state.min_soc
    max_e = capacity * state.max_soc
    # DC energy that must remain stored to deliver the reserve on demand.
    reserve_energy = reserve.reserve_kw * reserve.duration_h / spec.discharge_eff
    floor_e = min_e + reserve_energy
    if floor_e > max_e + EPS:
        return DispatchPlan(
            False,
            (),
            state,
            0.0,
            0.0,
            "reserve energy exceeds usable SOC window",
        )

    e = min(max(state.energy_kwh, min_e), max_e)
    segments: list[PlanSegment] = []
    total_throughput = 0.0
    total_cost = 0.0
    feasible = True
    reason = "ok"
    if e < floor_e - EPS:
        feasible = False
        reason = "initial SOC below reserve energy floor"

    for i, (p_req, dt, temp) in enumerate(zip(requests, durations, ambient)):
        derate = derating.factor(temp)
        p_temp = derate * spec.max_discharge_kw
        c_temp = derate * spec.max_charge_kw

        # The reserve must remain deliverable on top of any scheduled export.
        p_reserve = max(0.0, p_temp - reserve.reserve_kw)
        if p_temp + EPS < reserve.reserve_kw:
            feasible = False
            reason = f"period {i}: reserve power unavailable at {temp:g} C"

        if p_req >= 0.0:
            # Discharge: grid export drains the battery through efficiency.
            p_energy = max(0.0, (e - floor_e) * spec.discharge_eff / dt)
            actual = min(p_req, p_reserve, p_energy)
            energy_change = -actual * dt / spec.discharge_eff
            throughput = -energy_change
            if actual >= p_req - EPS:
                binding = "none"
            elif p_energy <= p_reserve + EPS and p_energy < p_req - EPS:
                binding = "soc_lower_limit"
            elif reserve.reserve_kw > EPS and p_reserve < p_temp - EPS:
                binding = "reserve_holdback"
            elif derate < 1.0 - EPS:
                binding = "temperature_derate"
            else:
                binding = "power_limit"
        else:
            # Charge: grid import fills the battery through efficiency.
            c_req = -p_req
            c_energy = max(0.0, (max_e - e) / (dt * spec.charge_eff))
            actual_c = min(c_req, c_temp, c_energy)
            actual = -actual_c
            energy_change = actual_c * dt * spec.charge_eff
            throughput = energy_change
            if actual_c >= c_req - EPS:
                binding = "none"
            elif c_energy <= c_temp + EPS and c_energy < c_req - EPS:
                binding = "soc_upper_limit"
            elif derate < 1.0 - EPS:
                binding = "temperature_derate"
            else:
                binding = "power_limit"

        e += energy_change
        e = min(max(e, floor_e), max_e)
        cost = throughput * spec.degradation_cost_per_kwh
        total_throughput += throughput
        total_cost += cost
        segments.append(
            PlanSegment(
                period=i,
                requested_kw=p_req,
                actual_kw=actual,
                duration_h=dt,
                energy_change_kwh=energy_change,
                end_soc=e / capacity,
                throughput_kwh=throughput,
                degradation_cost=cost,
                binding_constraint=binding,
            )
        )

    final_state = StorageState(e, capacity, state.min_soc, state.max_soc) if feasible else state
    return DispatchPlan(
        feasible=feasible,
        segments=tuple(segments),
        final_state=final_state,
        total_throughput_kwh=total_throughput,
        total_cost=total_cost,
        reason=reason,
    )


def commit_plan(state: StorageState, plan: DispatchPlan) -> StorageState:
    """Apply a feasible plan atomically.

    Raises ``ValueError`` for an infeasible plan so a failed or cancelled
    plan can never leave a partial energy change behind.
    """

    if not plan.feasible:
        raise ValueError(f"cannot commit dispatch plan: {plan.reason}")
    if plan.final_state is None:
        raise ValueError("dispatch plan carries no committed state")
    return plan.final_state
