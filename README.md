# AdaFrenetic + CommonRoad Bridge

A physics-based simulation bridge that connects [AdaFrenetic](https://github.com/ERATOMMSD/frenetic-sbst21) — a genetic algorithm that evolves challenging roads for autonomous driving testing — with the [CommonRoad](https://commonroad.in.tum.de/) Single-Track (ST) vehicle model. This replaces the BeamNG simulator dependency with an open-source, cross-platform alternative that runs entirely in Python.

---

## What This Is

AdaFrenetic is a test generator for autonomous driving. Its job is to evolve road geometries (sequences of curves) that cause a self-driving vehicle to fail — to drive out of its lane. Originally it was built for the [SBST 2022 Tool Competition](https://sbst22.github.io/tools/) and requires [BeamNG.tech](https://www.beamng.tech/), a commercial simulator that runs only on Windows.

This bridge solves that problem: it replaces BeamNG with CommonRoad's Single-Track vehicle dynamics model, making the full pipeline run on macOS and Linux with no commercial software required.

**The result**: a complete, self-contained test generation pipeline — AdaFrenetic evolves roads, CommonRoad simulates the vehicle, and at the end you get plots of the worst road found and charts showing how the genetic algorithm improved over time.

---

## Why CommonRoad Instead of BeamNG

| | BeamNG.tech | CommonRoad ST |
|---|---|---|
| **Platform** | Windows only | macOS, Linux, Windows |
| **License** | Commercial (free for research, registration required) | Open-source (BSD) |
| **Physics** | Full 3D soft-body, real-time rendering | Single-Track model: tire slip, lateral forces, yaw dynamics |
| **Requires GPU** | Yes (DirectX 11, 16 GB RAM) | No |
| **Speed** | ~1 test/5s (real-time sim) | ~50–500 tests/min |
| **Driver model** | Built-in AI | Pure Pursuit controller |
| **Integration** | BeamNGpy Python API | Direct Python in-process |

BeamNG is more realistic — it's a full 3D game engine with soft-body physics. But for the purpose of road evolution (finding curves that challenge a vehicle), the CommonRoad ST model is accurate enough. It captures the effects that matter: tire slip, lateral load transfer, understeer/oversteer, and yaw dynamics. The vehicle will go off-road on the same types of roads that would cause problems in a real car.

### The CommonRoad Single-Track Model

The ST model tracks 7 state variables at every timestep (50 Hz):

```
state = [x, y, δ, v, ψ, ψ̇, β]
         │  │  │  │  │  │   └─ slip angle at vehicle center of gravity
         │  │  │  │  │  └───── yaw rate (rad/s)
         │  │  │  │  └──────── yaw angle / heading (rad)
         │  │  │  └─────────── longitudinal velocity (m/s)
         │  │  └────────────── front wheel steering angle (rad)
         │  └───────────────── y position (m)
         └──────────────────── x position (m)

input  = [δ̇, aₓ]
          │   └── longitudinal acceleration (m/s²)
          └────── steering rate (rad/s)
```

Vehicle parameters come from `parameters_vehicle1()` (Ford Escort): wheelbase 2.39 m, max steering 0.91 rad, width 1.674 m. The dynamics are integrated with RK45 (Runge–Kutta 4/5) at each 20 ms step, with an Euler fallback if the integrator diverges.

### The Pure Pursuit Controller

The driver is a Pure Pursuit controller — a classical path-following algorithm used in real autonomous vehicles. At each step it:

1. Finds the closest point on the road centerline to the car
2. Projects a look-ahead point ahead along the centerline (`look_ahead = max(5.0, speed × 0.6)`)
3. Computes the angle `α` between the car's heading and the direction to the look-ahead point
4. Returns the steering angle: `δ = arctan(2 · L · sin(α) / look_ahead)`

The speed-adaptive look-ahead is important: at high speed it looks further ahead (more time to react), which mimics human driving and means the driver is genuinely harder to fool at moderate curves but fails at sharp ones.

---

## Modifications to AdaFrenetic

The original AdaFrenetic code is preserved in `../adafrenetic-sbst22/`. All modifications are applied automatically by `patch_adafrenetic.sh` and do not change the algorithm logic.

### Patch 1 — Random Phase Duration (most important)

**Original code** (`adaptive_random_frenet_generator.py`):
```python
class AdaFrenetic(CustomFrenetGenerator):
    def __init__(self, time_budget=None, executor=None, map_size=None):
        super().__init__(..., random_budget=3600, ...)
```

**Patched code**:
```python
class AdaFrenetic(CustomFrenetGenerator):
    def __init__(self, time_budget=None, executor=None, map_size=None):
        budget = 3600
        if executor and hasattr(executor, 'time_budget') and executor.time_budget.time_budget:
            budget = int(executor.time_budget.time_budget * 0.2)
        super().__init__(..., random_budget=budget, ...)
```

**Why this matters**: `random_budget` controls when the algorithm switches from random road generation to guided mutation. The random phase runs while `remaining_time > total_budget - random_budget`. With `random_budget=3600` (1 hour) and a `TIME_BUDGET=500`, the condition is always `remaining > 500 - 3600 = -3100` — which is always true. The algorithm **never reached the mutation phase** with any budget under one hour.

The fix sets `random_budget = 20% of TIME_BUDGET`, so with a 500 s budget: 100 s of random exploration, then 400 s of genetic evolution.

**Does it affect algorithm quality?** No. The 80/20 split is actually what the original authors intended — the comment in `CustomFrenetGenerator` says _"Spending 20% of the time on random generation"_. The hardcoded `3600` was designed for multi-hour competition runs. Scaling it proportionally restores the intended behavior.

### Patch 2 — pandas 2.0 Compatibility

```python
# Before (crashes on pandas 2.0):
self.df.at[parent.index, 'visited'] = True

# After:
self.df.loc[parent.index, 'visited'] = True
```

The `.at` accessor with an Index object (not a scalar) is rejected in pandas 2.0. Using `.loc` is equivalent — it sets the same value in the same row, just with the correct API. No logic change.

### Patch 3 — DataFrame append (pandas 2.0)

```python
# Before (deprecated in pandas 1.4, removed in 2.0):
self.df = self.df.append(info, ignore_index=True)

# After:
self.df = pd.concat([self.df, pd.DataFrame([info])], ignore_index=True)
```

### Patch 4 — Missing directory handling

```python
# Before (crashes if simulations/beamng_executor/ doesn't exist):
last_file = sorted(Path('simulations/beamng_executor').iterdir(), ...)[-1]
info['simulation_file'] = last_file.name

# After:
try:
    last_file = sorted(Path('simulations/beamng_executor').iterdir(), ...)[-1]
    info['simulation_file'] = last_file.name
except (FileNotFoundError, IndexError):
    info['simulation_file'] = None
```

### Patch 5 — numpy 2.0 Compatibility

```python
# Before: np.NaN  (removed in numpy 2.0)
# After:  np.nan
```

### Patch 6 — Performance (sleep removal)

Removes `time.sleep(5)` in the mock executor and `sleep(10)` in the generator — not relevant when using the CommonRoad executor, but left in to avoid wasting time if mock is used for testing.

### Patch 7 — Road length scaling bug (important)

**Original code**:
```python
self.max_length = 30       # hardcoded
self.frenet_step = 10
self.number_of_points = min(map_size // self.frenet_step, self.max_length)
```

**Patched code**:
```python
self.frenet_step = 10
self.max_length = map_size // self.frenet_step  # scales with map
self.number_of_points = self.max_length
```

**Why this matters**: `max_length=30` was hardcoded for the original 200m competition maps. With larger maps (e.g. `MAP_SIZE=1000`) the formula `min(1000//10, 30) = 30` always hit the cap — roads were still only 30 kappa points × 10m = **300m arc length maximum**, meaning their bounding box was typically only 100–150m regardless of how large the map was. The car was effectively driving in a small corner of a huge empty map.

The fix scales `max_length` with the map: for `MAP_SIZE=1000`, roads now get up to 100 kappa points × 10m = **1000m arc length**, generating complex roads that actually use the available space.

### New: CommonRoad Executor (`code_pipeline/commonroad_executor.py`)

The core addition: a drop-in replacement for `BeamngExecutor` that implements `AbstractTestExecutor` and returns real physics-based results.

<details>
<summary>View: CommonRoadExecutor._execute()</summary>

```python
def _execute(self, the_test):
    super()._execute(the_test)

    # Use AdaFrenetic's already-interpolated centerline
    interp = the_test.interpolated_points
    centerline = np.array([(p[0], p[1]) for p in interp])
    centerline_line = LineString(centerline)

    # Initialize ST vehicle at road start, aligned to road direction
    start_dir = centerline[1] - centerline[0]
    vehicle = STVehicle(
        x=centerline[0][0], y=centerline[0][1],
        heading=np.arctan2(start_dir[1], start_dir[0]),
        speed_ms=self.speed_ms
    )

    dt = 0.02       # 50 Hz simulation
    half_road = ROAD_WIDTH / 2.0   # 4.0 m
    car_half_width = 0.84          # Ford Escort: 1.674 m / 2

    execution_data = []
    oob_counter = 0
    in_oob = False
    max_oob_pct = 0.0

    for step in range(int(30.0 / dt)):   # max 30 s per road
        deviation = Point(vehicle.x, vehicle.y).distance(centerline_line)
        oob_distance = half_road - deviation   # positive = safe, negative = off-road

        # Approximate fraction of car body outside road
        edge_dist = half_road - deviation
        if edge_dist >= car_half_width:
            oob_pct = 0.0
        elif edge_dist <= -car_half_width:
            oob_pct = 1.0
        else:
            oob_pct = 0.5 - (edge_dist / (2.0 * car_half_width))

        is_oob = oob_pct > self.oob_tolerance
        if is_oob and not in_oob:
            oob_counter += 1
            in_oob = True
        elif not is_oob:
            in_oob = False

        execution_data.append(SimulationDataRecord(
            timer=step * dt,
            pos=(vehicle.x, vehicle.y, 0.0),
            ...
            oob_distance=oob_distance,   # ← this feeds AdaFrenetic's fitness
            oob_percentage=oob_pct,
            is_oob=is_oob,
            oob_counter=oob_counter,
            max_oob_percentage=max_oob_pct,
        ))

        # Pure Pursuit → ST model step
        look_ahead = max(5.0, vehicle.speed * 0.6)
        steering = pure_pursuit_steering(vehicle.x, vehicle.y, vehicle.heading,
                                         centerline, vehicle.wheelbase,
                                         look_ahead, vehicle.max_steering)
        vehicle.step(steering, self.speed_ms, dt)

    outcome = "FAIL" if oob_counter > 0 else "PASS"
    return outcome, description, execution_data
```

</details>

The key value AdaFrenetic reads from `execution_data` is:

```python
min_oob_distance = min(record.oob_distance for record in execution_data)
```

Roads with `min_oob_distance < -0.5` are selected as parents for mutation. A road with `-2.0` was significantly off-road — the genetic algorithm will prioritize mutating it to find even harder variants.

---

## How the Genetic Algorithm Works (Full Flow)

When you run `./run_experiment.sh`, the following loop runs for `TIME_BUDGET` seconds:

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Random Exploration (first 20% of TIME_BUDGET)     │
│                                                             │
│  LOOP:                                                      │
│    1. Generate random kappa values (road curvatures)        │
│       ~25 floats in [-0.05, 0.05]                          │
│    2. Convert kappas → (x, y) road via Frenet frame        │
│    3. Validate: inside map? no self-intersection?           │
│    4. Simulate with CommonRoad ST + Pure Pursuit            │
│    5. Record min_oob_distance in population DataFrame       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Guided Mutation (remaining 80%)                   │
│                                                             │
│  LOOP:                                                      │
│    1. Find best parent: unvisited road with                 │
│       min_oob_distance < -0.5, sorted by fitness           │
│    2a. If parent outcome == PASS → apply mutations:         │
│        ├─ add 1–5 kappas at end (extend road)              │
│        ├─ remove 1–5 kappas (shorten road)                 │
│        ├─ remove from front / from tail                     │
│        ├─ increase all kappas 10–20% (sharpen curves)      │
│        └─ randomly modify 1–5 kappas                       │
│    2b. If parent outcome == FAIL → apply mutations:         │
│        ├─ reverse road (drive it backwards)                 │
│        ├─ reverse kappas                                    │
│        ├─ split-and-swap (splice two halves)               │
│        └─ flip sign of all kappas (mirror curves)          │
│    3. If no good parent → fall back to random generation    │
│    4. Every 40 mutations → crossover phase:                 │
│        ├─ chromosome crossover (gene-by-gene mix)           │
│        └─ single-point crossover (splice two parents)      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Post-hoc Analysis (bridge_simulator.py)                    │
│    - Re-simulate all generated roads                        │
│    - Write summary CSV                                      │
│    - Plot worst road (hardest found)                        │
│    - Plot evolution chart (4 metrics over time)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
commonroad-bridge/
├── run_experiment.sh          ← main entry point (edit config here)
├── patch_adafrenetic.sh       ← auto-patches AdaFrenetic for compatibility
├── bridge_simulator.py        ← post-hoc ST simulation + CSV + plots
├── commonroad_executor.py     ← CommonRoad executor (copied into AdaFrenetic)
├── requirements.txt           ← bridge venv dependencies
└── results/
    └── run_TIMESTAMP_vSPEED_tTOL/
        ├── summary.csv           ← aggregated metrics
        ├── worst_road.png        ← hardest road + car trajectory
        ├── evolution.png         ← genetic algorithm evolution charts
        └── adafrenetic_output.log

../adafrenetic-sbst22/         ← original AdaFrenetic repo (patched in-place)
```

---

## Installation

### Prerequisites

- **macOS or Linux**
- **Python 3.10** (CommonRoad's dependency chain requires ≤ 3.13; 3.14 does not work)
- **Homebrew** (macOS) or system Python 3.10

Check your Python versions:
```bash
python3 --version          # system default
/opt/homebrew/bin/python3.10 --version   # Homebrew 3.10 (macOS)
```

Install Python 3.10 on macOS if needed:
```bash
brew install python@3.10
```

### Step 1: Clone AdaFrenetic

```bash
git clone https://github.com/ERATOMMSD/frenetic-sbst21 adafrenetic-sbst22
# Note: the SBST22 version used here is the same codebase
```

Or clone the SBST 2022 fork directly if available.

### Step 2: Create AdaFrenetic venv (Python 3.10)

```bash
cd adafrenetic-sbst22
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy scipy shapely click pandas matplotlib commonroad-vehicle-models descartes
deactivate
```

> **Important**: Use Python 3.10 specifically. The `commonroad-vehicle-models` package requires `antlr4-python3-runtime`, which does not support Python 3.14.

### Step 3: Create bridge venv

```bash
cd commonroad-bridge
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

### Step 4: Run

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

The script will:
1. Patch AdaFrenetic automatically (only on first run — creates a `.patched` marker)
2. Run AdaFrenetic with the CommonRoad executor
3. Run the bridge simulator for post-hoc analysis and plots

---

## Configuration

Edit the top of `run_experiment.sh`:

```bash
# ===== CONFIGURATION =========================================================

TIME_BUDGET=500    # Total time for road generation (seconds)
                   # Recommended: 300–3600. Below 60s the algorithm barely
                   # has time to find interesting roads.

MAP_SIZE=500       # Square map side length (meters)
                   # Roads must fit within this boundary.

SPEED=80           # Vehicle speed (km/h)
                   # Higher speed → more FAILs on curves
                   # Recommended range: 50–120

OOB_TOLERANCE=0.85 # Fraction of car body outside road to count as OOB
                   # 0.85 = 85% of car must be outside before it counts
                   # Lower values → more sensitive OOB detection
                   # Recommended range: 0.6–0.95
```

### Effect of parameters on results

| Parameter | Lower value | Higher value |
|---|---|---|
| `TIME_BUDGET` | Fewer roads, less evolution | More generations, better optima |
| `SPEED` | Car handles most roads (few FAILs) | More failures, harder mutation parents found |
| `OOB_TOLERANCE` | Stricter OOB detection (more FAILs) | Lenient OOB detection (fewer FAILs) |
| `MAP_SIZE` | Shorter roads (less complex curves) | Longer roads (more complexity possible) |

A good starting configuration for seeing mutations in action: `TIME_BUDGET=300`, `SPEED=100`, `OOB_TOLERANCE=0.85`.

---

## Output

Each run produces a dedicated folder:

```
results/run_20260322_210603_v100_t0.85/
```

### `summary.csv`

| Column | Description |
|---|---|
| `nr scenarii valide` | Roads that passed geometry validation |
| `nr scenarii invalide` | Roads rejected (out of map, self-intersecting, too short) |
| `iesiri de pe banda OBE` | Total out-of-bounds events across all valid roads |
| `eficienta/timp de generare (s)` | Wall-clock time for post-hoc simulation |
| `Valoare de fitness` | Average `min_oob_distance` across valid roads (lower = harder) |
| `rezultat final` | `N SUCCESS / M FAIL` summary |

### `worst_road.png`

The road with the lowest `min_oob_distance` (hardest for the vehicle). Shows:
- Road surface (grey fill) with centerline
- AdaFrenetic control points (blue squares)
- Car trajectory colored by deviation from center (green=safe, red=far off-road)
- OOB positions (red scatter)
- Start (green triangle) and end (red triangle) markers

### `evolution.png`

Four subplots showing how the genetic algorithm improved over the run:

1. **min_oob_distance per test** — each dot is one road. The red line tracks the running best (most negative = hardest road found so far). Steps downward show the algorithm finding better parents.
2. **Cumulative FAILs** — staircase pattern shows when failures are found. Accelerates during mutation phase.
3. **Generation method** — color-coded by strategy (random, reverse, flip sign, split & swap, crossover). The left portion is all random; mutations appear in the right portion.
4. **Accumulated negative OOB** — severity of lane departure per road (taller = more time spent off-road).

---

## Links

- **AdaFrenetic (Frenetic SBST21)**: https://github.com/ERATOMMSD/frenetic-sbst21
- **SBST 2022 Tool Competition**: https://sbst22.github.io/tools/
- **CommonRoad**: https://commonroad.in.tum.de/
- **CommonRoad Vehicle Models (pip)**: `pip install commonroad-vehicle-models`
- **BeamNG.tech (original simulator)**: https://www.beamng.tech/
- **BeamNGpy API**: https://github.com/BeamNG/BeamNGpy

---

<details>
<summary>Code: STVehicle — CommonRoad Single-Track wrapper</summary>

```python
class STVehicle:
    """
    State: [x, y, delta, v, psi, psi_dot, beta]
    Input: [delta_dot, a_x]
    """
    def __init__(self, x, y, heading, speed_ms):
        self.p = parameters_vehicle1()           # Ford Escort parameters
        self.wheelbase = self.p.a + self.p.b    # 2.39 m
        self.max_steering = self.p.steering.max  # 0.91 rad
        self.state = np.array([x, y, 0.0, speed_ms, heading, 0.0, 0.0])

    def step(self, desired_steering, target_speed_ms, dt):
        desired_steering = np.clip(desired_steering, -self.max_steering, self.max_steering)
        delta_dot = np.clip((desired_steering - self.state[2]) / dt, -0.4, 0.4)
        a_x = 2.0 * (target_speed_ms - self.state[3])
        u = [delta_dot, a_x]

        sol = solve_ivp(
            lambda t, x: vehicle_dynamics_st(x, u, self.p),
            [0, dt], self.state, method='RK45',
            max_step=dt / 2, rtol=1e-6, atol=1e-6
        )
        if sol.success:
            self.state = sol.y[:, -1]
        else:                            # Euler fallback if RK45 diverges
            dx = vehicle_dynamics_st(self.state, u, self.p)
            self.state = self.state + np.array(dx) * dt

        self.state[3] = max(self.state[3], 0.1)   # keep velocity positive
```

</details>

<details>
<summary>Code: Pure Pursuit controller</summary>

```python
def pure_pursuit_steering(x, y, heading, road_points, wheelbase, look_ahead, max_steering):
    car_pos = np.array([x, y])
    distances = np.linalg.norm(road_points - car_pos, axis=1)
    closest_idx = np.argmin(distances)

    # Walk forward along road until look-ahead distance is reached
    target_idx = closest_idx
    for i in range(closest_idx, len(road_points)):
        if np.linalg.norm(road_points[i] - car_pos) >= look_ahead:
            target_idx = i
            break
    else:
        target_idx = len(road_points) - 1

    target = road_points[target_idx]
    dx, dy = target[0] - x, target[1] - y
    alpha = np.arctan2(dy, dx) - heading
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]

    steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), look_ahead)
    return np.clip(steering, -max_steering, max_steering)
```

</details>

<details>
<summary>Code: AdaFrenetic mutation strategies</summary>

```python
# When a road PASSED — try to make it harder by modifying curvatures
def mutate_passed_test(self, parent, parent_info):
    kappa_mutations = [
        ('add 1 to 5 kappas at the end',     self.add_kappas),
        ('randomly remove 1 to 5 kappas',    self.randomly_remove_kappas),
        ('remove 1 to 5 kappas from front',  lambda ks: ks[random.randint(1, 5):]),
        ('remove 1 to 5 kappas from tail',   lambda ks: ks[:-random.randint(1, 5)]),
        ('increase all kappas 10~20%',        self.increase_kappas),
        ('randomly modify 1 to 5 kappas',    self.random_modification),
    ]
    self.perform_kappa_mutations(kappa_mutations, parent, parent_info)

# When a road FAILED — explore structural variants of the failure
def mutate_failed_test(self, parent, parent_info):
    # Try reversed road first
    self.execute_test(road_points[::-1], method='reversed road', ...)

    kappa_mutations = [
        ('reverse kappas',          lambda ks: ks[::-1]),
        ('split and swap kappas',   lambda ks: ks[len(ks)//2:] + ks[:len(ks)//2]),
        ('flip sign kappas',        lambda ks: list(map(lambda x: x * -1.0, ks))),
    ]
    self.perform_kappa_mutations(kappa_mutations, parent, parent_info, ...)
```

</details>

<details>
<summary>Code: OOB distance → fitness feedback loop</summary>

```python
# In base_generator.py — how AdaFrenetic extracts fitness from simulation data
metrics = ['oob_distance', ...]
functions = [('max', np.max), ('min', np.min), ('mean', np.mean), ('avg', np.average)]
for metric, (name, func) in itertools.product(metrics, functions):
    metric_data = [getattr(record, metric) for record in execution_data]
    info[f'{name}_{metric}'] = func(metric_data)
# info['min_oob_distance'] is now set

# In generate_mutants() — parent selection
parent = self.df[
    ((self.df.outcome == 'PASS') | (self.df.outcome == 'FAIL'))
    & (~self.df.visited)
    & (self.df.min_oob_distance < -0.5)   # threshold
].sort_values('min_oob_distance', ascending=True).head(1)
```

The key insight: `oob_distance = ROAD_WIDTH/2 - deviation_from_center`. Positive means the car is safely inside the road. Negative means the car center crossed the lane edge. The most negative value in a run is `min_oob_distance` — the genetic algorithm selects roads where this goes below -0.5 as parents for the next generation.

</details>
