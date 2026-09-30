from __future__ import annotations
from dataclasses import dataclass, field

# --- Legacy state (kept for backward compatibility) -------------------------

@dataclass(frozen=True)
class StorageState:
    energy_kwh: float
    capacity_kwh: float
    min_soc: float=0.1
    max_soc: float=0.95

def soc(state: StorageState) -> float:
    return state.energy_kwh/state.capacity_kwh

# --- Extended configuration -------------------------------------------------

@dataclass(frozen=True)
class StorageConfig:
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    min_soc: float=0.1
    max_soc: float=0.95
    charge_eff: float=0.95
    discharge_eff: float=0.95
    # Mandatory contingency reserve (energy kept above min_soc at all times),
    # expressed as a fraction of nameplate capacity.
    reserve_soc: float=0.0
    # Marginal throughput degradation cost per kWh cycled through the battery.
    # May also be derived from replacement cost / cycle life via
    # degradation_cost_per_kwh().
    degradation_cost_per_kwh: float=0.0
    # Piecewise-linear temperature power derating curve: sorted list of
    # (ambient_temperature_c, available_fraction) points, clamped at the ends.
    temp_derate_curve: tuple[tuple[float,float],...]=()

    def __post_init__(self):
        if self.capacity_kwh<=0: raise ValueError("capacity_kwh must be positive")
        if self.max_charge_kw<0 or self.max_discharge_kw<0:
            raise ValueError("power ratings must be non-negative")
        if not 0<=self.min_soc<self.max_soc<=1: raise ValueError("invalid soc limits")
        if not 0<self.charge_eff<=1 or not 0<self.discharge_eff<=1:
            raise ValueError("efficiencies must lie in (0,1]")
        if not 0<=self.reserve_soc<=1: raise ValueError("reserve_soc must lie in [0,1]")
        if self.min_soc+self.reserve_soc>self.max_soc+1e-12:
            raise ValueError("reserve_soc does not fit between soc limits")
        if self.degradation_cost_per_kwh<0: raise ValueError("degradation cost must be non-negative")
        t=[p[0] for p in self.temp_derate_curve]
        if t!=sorted(t): raise ValueError("temp derate curve must be temperature-sorted")
        for _,f in self.temp_derate_curve:
            if not 0<=f<=1: raise ValueError("derate fractions must lie in [0,1]")

    @property
    def reserve_energy_kwh(self) -> float:
        return self.capacity_kwh*self.reserve_soc

def degradation_cost_per_kwh(replacement_cost: float, lifetime_kwh_throughput: float) -> float:
    """Marginal wear cost: battery replacement cost divided by total lifetime
    energy throughput (kWh cycled through the battery)."""
    if replacement_cost<0 or lifetime_kwh_throughput<=0:
        raise ValueError("invalid replacement cost or lifetime throughput")
    return replacement_cost/lifetime_kwh_throughput

def temperature_derate(config: StorageConfig, ambient_c: float) -> float:
    """Fraction of nameplate charge/discharge power available at the given
    ambient temperature, interpolated on the configured derating curve."""
    curve=config.temp_derate_curve
    if not curve: return 1.0
    if ambient_c<=curve[0][0]: return curve[0][1]
    if ambient_c>=curve[-1][0]: return curve[-1][1]
    for (t0,f0),(t1,f1) in zip(curve,curve[1:]):
        if t0<=ambient_c<=t1:
            if t1==t0: return f1
            return f0+(f1-f0)*(ambient_c-t0)/(t1-t0)
    return 1.0

# --- Multi-period plan primitives ------------------------------------------

@dataclass(frozen=True)
class SegmentRequest:
    duration_h: float
    requested_kw: float
    price_per_kwh: float=0.0          # grid-terminal energy price (+ buy / - sell)
    ambient_c: float=25.0
    # Reserve required at the end of this segment only; defaults to the
    # configuration reserve which applies throughout.
    reserve_soc: float|None=None
    label: str=""

    def __post_init__(self):
        if self.duration_h<=0: raise ValueError("duration_h must be positive")
        if self.reserve_soc is not None and not 0<=self.reserve_soc<=1:
            raise ValueError("reserve_soc must lie in [0,1]")

@dataclass(frozen=True)
class SegmentResult:
    index: int
    label: str
    duration_h: float
    requested_kw: float
    delivered_kw: float               # grid-terminal power, + discharge / - charge
    energy_kwh: float                 # grid-terminal energy, signed like power
    battery_kwh: float                # energy actually cycled through the battery (+)
    soc_start: float
    soc_end: float
    ambient_c: float
    temp_derate: float
    energy_cost: float                # cost (+) / revenue (-) at grid terminal
    degradation_cost: float           # throughput wear cost
    total_cost: float
    # Which constraint, if any, capped the delivered power relative to request:
    # "none" | "power" | "temperature" | "soc_upper" | "soc_lower" | "reserve"
    binding_constraint: str
    reason: str                       # human-readable explanation

@dataclass(frozen=True)
class ScheduleReport:
    segments: tuple[SegmentResult,...]
    initial_energy_kwh: float
    final_energy_kwh: float
    initial_soc: float
    final_soc: float
    energy_cost: float
    degradation_cost: float
    total_cost: float
    committed: bool
    failure: str|None=None

def _dispatch_segment(
    config: StorageConfig,
    energy_kwh: float,
    req: SegmentRequest,
    index: int,
    strict: bool,
) -> tuple[SegmentResult,float,str|None]:
    """Dispatch one segment against a starting energy. Returns the result,
    the resulting stored energy, and a failure message (strict mode only)."""
    cap=config.capacity_kwh
    e_min=cap*config.min_soc
    e_max=cap*config.max_soc
    reserve_soc=config.reserve_soc if req.reserve_soc is None else req.reserve_soc
    e_floor=min(e_max,e_min+cap*reserve_soc)
    dt=req.duration_h
    derate=temperature_derate(config,req.ambient_c)
    eff_c,eff_d=config.charge_eff,config.discharge_eff
    soc_start=energy_kwh/cap

    failure: str|None=None
    if req.requested_kw>=0.0:
        # Discharge: battery loses delivered / discharge_eff
        head_power=config.max_discharge_kw*derate
        head_energy=max(0.0,(energy_kwh-e_floor)*eff_d/dt)
        deliverable=min(head_power,head_energy)
        delivered=min(req.requested_kw,max(0.0,deliverable))
        battery=delivered*dt/eff_d
        new_energy=energy_kwh-battery
        if req.requested_kw>0:
            if delivered+1e-9<req.requested_kw:
                if head_energy+1e-9<head_power:
                    if reserve_soc>1e-12:
                        binding="reserve"
                        reason=(f"discharge limited to {delivered:.3f} kW: contingency reserve "
                                f"floor {reserve_soc:.3f} SOC above min ({e_floor:.2f} kWh) binds")
                    else:
                        binding="soc_lower"
                        reason=(f"discharge limited to {delivered:.3f} kW: lower SOC limit "
                                f"{config.min_soc:.3f} ({e_min:.2f} kWh) reached")
                else:
                    binding="temperature" if derate<1.0-1e-12 else "power"
                    if binding=="temperature":
                        reason=(f"discharge limited to {delivered:.3f} kW: temperature derate "
                                f"{derate:.3f} at {req.ambient_c:.1f} C")
                    else:
                        reason=(f"discharge limited to {delivered:.3f} kW: discharge power rating "
                                f"{config.max_discharge_kw:.1f} kW binds")
            else:
                binding="none"
                reason=f"full request of {req.requested_kw:.3f} kW dispatched"
        else:
            binding="none"; reason="zero discharge request; idle"
    else:
        # Charge: grid pushes -requested_kw, battery gains that * charge_eff
        requested_charge=-req.requested_kw
        head_power=config.max_charge_kw*derate
        head_energy=max(0.0,(e_max-energy_kwh)/(eff_c*dt))
        absorbable=min(head_power,head_energy)
        charge=min(requested_charge,max(0.0,absorbable))
        delivered=-charge
        battery=charge*dt*eff_c
        new_energy=energy_kwh+battery
        if requested_charge>0:
            if charge+1e-9<requested_charge:
                if head_energy+1e-9<head_power:
                    binding="soc_upper"
                    reason=(f"charge limited to {charge:.3f} kW: upper SOC limit "
                            f"{config.max_soc:.3f} ({e_max:.2f} kWh) reached")
                else:
                    binding="temperature" if derate<1.0-1e-12 else "power"
                    if binding=="temperature":
                        reason=(f"charge limited to {charge:.3f} kW: temperature derate "
                                f"{derate:.3f} at {req.ambient_c:.1f} C")
                    else:
                        reason=(f"charge limited to {charge:.3f} kW: charge power rating "
                                f"{config.max_charge_kw:.1f} kW binds")
            else:
                binding="none"
                reason=f"full request of {requested_charge:.3f} kW dispatched"
        else:
            binding="none"; reason="zero charge request; idle"

    if strict and binding!="none" and abs(req.requested_kw)>1e-12:
        failure=(f"segment {index}"
                 +(f" ({req.label})" if req.label else "")
                 +f": {reason}")

    # Clamp into limits to absorb floating-point noise.
    new_energy=min(max(new_energy,e_floor if req.requested_kw>=0 else e_min),e_max)

    grid_energy=delivered*dt
    # Positive cost = payment (charging buys energy); discharge earns revenue.
    energy_cost=-grid_energy*req.price_per_kwh
    deg_cost=battery*config.degradation_cost_per_kwh
    result=SegmentResult(
        index=index,
        label=req.label,
        duration_h=dt,
        requested_kw=req.requested_kw,
        delivered_kw=delivered,
        energy_kwh=grid_energy,
        battery_kwh=battery,
        soc_start=soc_start,
        soc_end=new_energy/cap,
        ambient_c=req.ambient_c,
        temp_derate=derate,
        energy_cost=energy_cost,
        degradation_cost=deg_cost,
        total_cost=energy_cost+deg_cost,
        binding_constraint=binding,
        reason=reason,
    )
    return result,new_energy,failure

def _feasible_start(config: StorageConfig, energy_kwh: float) -> str|None:
    cap=config.capacity_kwh
    e_min=cap*config.min_soc; e_max=cap*config.max_soc
    if not e_min-1e-9*cap<=energy_kwh<=e_max+1e-9*cap:
        return (f"initial energy {energy_kwh:.2f} kWh outside SOC band "
                f"[{e_min:.2f}, {e_max:.2f}] kWh")
    return None

def plan_multi_period(
    config: StorageConfig,
    initial_energy_kwh: float,
    segments: list[SegmentRequest],
    *,
    strict: bool=False,
) -> ScheduleReport:
    """Evaluate a multi-period dispatch plan. The plan is evaluated
    end-to-end without mutating any caller state: a failing strict plan is
    rolled back (no committed energy change). With strict=False partial
    dispatch is allowed and every capping constraint is reported per segment.
    """
    if not segments: raise ValueError("at least one segment is required")
    bad=_feasible_start(config,initial_energy_kwh)
    if bad is not None: raise ValueError(bad)

    cap=config.capacity_kwh
    e_min=cap*config.min_soc; e_max=cap*config.max_soc
    energy=min(max(initial_energy_kwh,e_min),e_max)
    results: list[SegmentResult]=[]
    failure: str|None=None
    for i,req in enumerate(segments):
        result,energy,fail=_dispatch_segment(config,energy,req,i,strict)
        results.append(result)
        if fail is not None and failure is None: failure=fail
    committed=failure is None
    if failure is not None:
        # Roll back: report the attempted dispatch but mark it not committed.
        energy=initial_energy_kwh
    return ScheduleReport(
        segments=tuple(results),
        initial_energy_kwh=initial_energy_kwh,
        final_energy_kwh=energy if committed else initial_energy_kwh,
        initial_soc=initial_energy_kwh/cap,
        final_soc=(energy if committed else initial_energy_kwh)/cap,
        energy_cost=sum(r.energy_cost for r in results) if committed else 0.0,
        degradation_cost=sum(r.degradation_cost for r in results) if committed else 0.0,
        total_cost=sum(r.total_cost for r in results) if committed else 0.0,
        committed=committed,
        failure=failure,
    )

class StorageScheduler:
    """Stateful wrapper around plan_multi_period. Plans are only committed
    atomically; cancellation or a failed strict plan leaves stored energy
    exactly as it was."""

    def __init__(self, config: StorageConfig, initial_energy_kwh: float):
        bad=_feasible_start(config,initial_energy_kwh)
        if bad is not None: raise ValueError(bad)
        self.config=config
        self.energy_kwh=min(max(initial_energy_kwh,
                                config.capacity_kwh*config.min_soc),
                            config.capacity_kwh*config.max_soc)

    @property
    def soc(self) -> float:
        return self.energy_kwh/self.config.capacity_kwh

    def preview(self, segments: list[SegmentRequest]) -> ScheduleReport:
        """Dry run against the current state; never changes stored energy."""
        return plan_multi_period(self.config,self.energy_kwh,segments,strict=False)

    def commit(self, segments: list[SegmentRequest], *, strict: bool=False) -> ScheduleReport:
        """Atomically dispatch the plan. On strict failure the state is
        untouched and the returned report is committed=False."""
        report=plan_multi_period(self.config,self.energy_kwh,segments,strict=strict)
        if report.committed:
            self.energy_kwh=report.final_energy_kwh
        return report

    def cancel(self, segments: list[SegmentRequest]|None=None, *, reason: str="cancelled by operator") -> ScheduleReport:
        """Explicitly abandon a plan. No segment runs; state is unchanged."""
        planned=segments or []
        if planned:
            dry=plan_multi_period(self.config,self.energy_kwh,planned,strict=False)
            results=tuple(SegmentResult(
                index=r.index,label=r.label,duration_h=r.duration_h,
                requested_kw=r.requested_kw,delivered_kw=0.0,energy_kwh=0.0,
                battery_kwh=0.0,soc_start=self.soc,soc_end=self.soc,
                ambient_c=r.ambient_c,temp_derate=r.temp_derate,
                energy_cost=0.0,degradation_cost=0.0,total_cost=0.0,
                binding_constraint="cancelled",
                reason=f"segment cancelled: {reason}",
            ) for r in dry.segments)
        else:
            results=()
        return ScheduleReport(
            segments=results,
            initial_energy_kwh=self.energy_kwh,
            final_energy_kwh=self.energy_kwh,
            initial_soc=self.soc,final_soc=self.soc,
            energy_cost=0.0,degradation_cost=0.0,total_cost=0.0,
            committed=False,failure=f"plan cancelled: {reason}",
        )

# --- Legacy single-segment dispatch (unchanged behaviour) -------------------

def dispatch(state: StorageState, requested_kw: float, duration_h: float, charge_eff: float=0.95, discharge_eff: float=0.95):
    if state.capacity_kwh <= 0 or duration_h < 0: raise ValueError("invalid storage state")
    if not 0 < charge_eff <= 1 or not 0 < discharge_eff <= 1: raise ValueError("invalid efficiency")
    min_e=state.capacity_kwh*state.min_soc; max_e=state.capacity_kwh*state.max_soc
    e=min(max(state.energy_kwh,min_e),max_e)
    if requested_kw >= 0:
        max_deliverable=max(0.0,(e-min_e)*discharge_eff/duration_h) if duration_h else 0.0
        actual=min(requested_kw,max_deliverable)
        new_e=e-actual*duration_h/discharge_eff
    else:
        charge_kw=-requested_kw
        max_absorb=max(0.0,(max_e-e)/(charge_eff*duration_h)) if duration_h else 0.0
        actual=-min(charge_kw,max_absorb)
        new_e=e+(-actual)*duration_h*charge_eff
    return actual,StorageState(new_e,state.capacity_kwh,state.min_soc,state.max_soc)
