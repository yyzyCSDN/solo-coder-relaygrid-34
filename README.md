# RelayGrid

RelayGrid is a deterministic Python toolkit for distribution-grid operations.
It covers feeder topology, protection, voltage assessment, DER islanding,
switching safety, outage accounting, crew logistics, asset thermal limits,
telemetry quality, storage dispatch, and restoration sequencing.

The package is self-contained and uses only the Python standard library.

## Install and run

```bash
python -m pip install -e ".[test]"
relaygrid demo
python -m pytest -q
```

The demo builds a four-bus feeder with two open ties, reports its energized
state, safely closes the sectionalizing branch, and emits a JSON summary.

## Public API

The top-level package exports the feeder data model and `GridSimulator`:

```python
from relaygrid import Branch, Bus, FeederSnapshot, GridSimulator, SwitchState
```

Domain modules remain independently importable so applications can combine
topology, protection, telemetry, restoration, reliability, planning, and field
operations without depending on a web framework or database.

## Multi-period storage dispatch

`relaygrid.storage` schedules battery segments across multiple periods with
charge/discharge efficiencies, a piecewise-linear temperature power-derating
curve, throughput-based cycle degradation cost, and a mandatory contingency
reserve (SOC floor above `min_soc`):

```python
from relaygrid.storage import (
    StorageConfig, StorageScheduler, SegmentRequest,
    degradation_cost_per_kwh,
)

cfg = StorageConfig(
    capacity_kwh=500, max_charge_kw=250, max_discharge_kw=250,
    charge_eff=0.95, discharge_eff=0.93,
    min_soc=0.10, max_soc=0.95, reserve_soc=0.15,
    degradation_cost_per_kwh=degradation_cost_per_kwh(30_000, 300_000),
    temp_derate_curve=((25, 1.0), (45, 0.4), (50, 0.2)),
)
sched = StorageScheduler(cfg, initial_energy_kwh=300)
report = sched.commit([
    SegmentRequest(1.0, +200, price_per_kwh=2.0, ambient_c=45),  # discharge
    SegmentRequest(2.0, -150, price_per_kwh=0.3, ambient_c=20),  # charge
])
```

Each `SegmentResult` reports delivered power, grid and battery energy, start/end
SOC, applied temperature derate, energy cost, degradation cost, total cost, and
the binding constraint (`power`, `temperature`, `soc_upper`, `soc_lower`,
`reserve`, or `none`) with a human-readable reason. SOC is never allowed outside
the configured band. `commit(..., strict=True)` rejects and atomically rolls
back any plan containing an unsatisfiable segment, `preview()` evaluates without
mutating state, and `cancel()` abandons a plan with zero delivered energy and
zero cost.


## License

RelayGrid is released under the MIT License.
