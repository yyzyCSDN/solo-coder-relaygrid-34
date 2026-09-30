from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from relaygrid.models import Branch, Bus, FeederSnapshot, SwitchState
from relaygrid.sim.engine import GridSimulator
from relaygrid.storage.dispatch import (
    BatterySpec,
    ReserveRequirement,
    StorageState,
    ThermalDerating,
    plan_dispatch,
    soc,
)


def storage_dispatch_summary() -> dict[str, object]:
    """Run a deterministic multi-period storage plan and return its summary.

    A 100 kWh battery with a 40 kW round-trip charge/discharge schedule is
    planned across four hours (midday charge, evening peak discharge). The
    plan accounts for temperature derating, a 20 kW contingency reserve that
    must stay available, round-trip efficiency and cycle throughput
    degradation cost. Every segment reports committed power, battery energy
    change, degradation cost and the constraint that bound the request.
    """
    state = StorageState(50.0, 100.0)
    spec = BatterySpec(
        max_charge_kw=50.0,
        max_discharge_kw=50.0,
        charge_eff=0.95,
        discharge_eff=0.95,
        degradation_cost_per_kwh=0.02,
    )
    reserve = ReserveRequirement(reserve_kw=20.0, duration_h=0.25)
    derating = ThermalDerating(full_output_c=25.0, derate_per_c=0.02, min_factor=0.5)
    plan = plan_dispatch(
        state,
        requested_kw=[-40.0, -40.0, 40.0, 40.0],
        duration_h=[1.0, 1.0, 1.0, 1.0],
        ambient_c=[30.0, 35.0, 38.0, 32.0],
        spec=spec,
        reserve=reserve,
        derating=derating,
    )
    return {
        "feasible": plan.feasible,
        "reason": plan.reason,
        "initial_soc": soc(state),
        "final_soc": soc(plan.final_state) if plan.final_state is not None else None,
        "total_throughput_kwh": round(plan.total_throughput_kwh, 4),
        "total_cost": round(plan.total_cost, 4),
        "segments": [
            {
                "period": s.period,
                "requested_kw": s.requested_kw,
                "actual_kw": round(s.actual_kw, 4),
                "duration_h": s.duration_h,
                "energy_change_kwh": round(s.energy_change_kwh, 4),
                "end_soc": round(s.end_soc, 4),
                "throughput_kwh": round(s.throughput_kwh, 4),
                "degradation_cost": round(s.degradation_cost, 4),
                "binding_constraint": s.binding_constraint,
            }
            for s in plan.segments
        ],
    }


def demo_summary() -> dict[str, object]:
    """Run a deterministic switching example and return its public summary."""
    buses = {bus_id: Bus(bus_id, 12.47) for bus_id in "ABCD"}
    branches = {
        "ab": Branch("ab", "A", "B", 0.1, 0.1, 300.0),
        "bc": Branch("bc", "B", "C", 0.1, 0.1, 300.0),
        "cd": Branch("cd", "C", "D", 0.1, 0.1, 300.0, True, False),
        "ad": Branch("ad", "A", "D", 0.1, 0.1, 300.0, True, False),
    }
    snapshot = FeederSnapshot(
        buses,
        branches,
        {"cd": SwitchState.OPEN, "ad": SwitchState.OPEN},
    )
    simulator = GridSimulator(snapshot)
    energized_before = simulator.energized_bus_count()
    simulator.operate_switch("cd", SwitchState.CLOSED)
    return {
        "energized_after": simulator.energized_bus_count(),
        "energized_before": energized_before,
        "events": len(simulator.log.events),
        "project": "RelayGrid",
        "switch": "cd",
        "switch_state": snapshot.switch_state["cd"].value,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relaygrid")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("demo", help="run a deterministic feeder demo")
    subcommands.add_parser("storage", help="run a multi-period storage dispatch demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        print(json.dumps(demo_summary(), sort_keys=True))
        return 0
    if args.command == "storage":
        print(json.dumps(storage_dispatch_summary(), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
