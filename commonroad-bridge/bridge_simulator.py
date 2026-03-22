#!/usr/bin/env python3
"""
CommonRoad Bridge Simulator for AdaFrenetic
============================================
Simulates a vehicle driving AdaFrenetic-generated roads using CommonRoad's
Single-Track (ST) dynamic vehicle model. The ST model adds tire slip, friction,
and yaw dynamics for more realistic behavior compared to a kinematic bicycle model.

Usage:
  python bridge_simulator.py --adafrenetic-dir ../adafrenetic-sbst22 --speed 80 --oob-tolerance 0.85
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime
from matplotlib.collections import LineCollection
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point

from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st


# =============================================================================
# Constants
# =============================================================================
ROAD_WIDTH = 8.0  # Standard SBST road width (meters)


# =============================================================================
# Road Processing (adapted from mini_simulator.py)
# =============================================================================

def smooth_road(points: np.ndarray, num_samples: int = 500) -> np.ndarray:
    """Interpolate sparse control points into a smooth B-spline curve."""
    if len(points) < 4:
        return points
    try:
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, len(points) - 1))
        u_fine = np.linspace(0, 1, num_samples)
        x_fine, y_fine = splev(u_fine, tck)
        return np.column_stack([x_fine, y_fine])
    except Exception:
        return points


def validate_road(points: np.ndarray, map_size: int = 200) -> Tuple[bool, str]:
    """Validate road: within map, no self-intersection, sufficient length."""
    if np.any(points < 0) or np.any(points > map_size):
        return False, "Outside map boundaries"
    try:
        line = LineString(points)
        if not line.is_simple:
            return False, "Self-intersecting"
    except Exception:
        return False, "Invalid geometry"
    total_length = sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points) - 1))
    if total_length < 20:
        return False, "Too short"
    return True, "Valid"


# =============================================================================
# Road Loading
# =============================================================================

def parse_adafrenetic_json(results_dir: str) -> List[np.ndarray]:
    """Load roads from AdaFrenetic JSON result files (most reliable source)."""
    roads = []
    subdirs = sorted(glob.glob(os.path.join(results_dir, '*')), key=os.path.getmtime, reverse=True)
    for subdir in subdirs:
        json_files = sorted(glob.glob(os.path.join(subdir, 'test.*.json')))
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                points = np.array(data['road_points'])
                if len(points) >= 3:
                    roads.append(points)
            except Exception:
                pass
        if roads:
            break
    if roads:
        print(f"  Loaded {len(roads)} roads from JSON in {results_dir}")
    return roads


def parse_adafrenetic_log(log_file: str) -> List[np.ndarray]:
    """Parse road points from AdaFrenetic console log."""
    roads = []
    with open(log_file, 'r') as f:
        content = f.read()
    pattern = r'Generated test using: \[(.*?)\]'
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        coord_pattern = r'(?:np\.float64\()?([\d.]+)\)?[,\s]+(?:np\.float64\()?([\d.]+)\)?'
        coords = re.findall(coord_pattern, match)
        if coords:
            points = np.array([(float(x), float(y)) for x, y in coords])
            roads.append(points)
    if roads:
        print(f"  Loaded {len(roads)} roads from {log_file}")
    return roads


def generate_sample_roads(num_roads: int = 20, map_size: int = 200) -> List[np.ndarray]:
    """Generate random roads for testing when no AdaFrenetic output is available."""
    np.random.seed(42)
    roads = []
    margin = 20
    for _ in range(num_roads):
        start_x = np.random.uniform(margin, map_size - margin)
        start_y = margin
        num_segments = np.random.randint(12, 22)
        step_length = 10.0
        points = [(start_x, start_y)]
        heading = np.pi / 2
        for _ in range(num_segments):
            kappa = np.random.uniform(-0.08, 0.08)
            heading += kappa * step_length
            new_x = points[-1][0] + step_length * np.cos(heading)
            new_y = points[-1][1] + step_length * np.sin(heading)
            points.append((new_x, new_y))
        roads.append(np.array(points))
    return roads


def load_roads(adafrenetic_dir: str, map_size: int) -> List[np.ndarray]:
    """Load roads: JSON results > log file > sample fallback."""
    roads = []

    results_dir = os.path.join(adafrenetic_dir, "results")
    if os.path.exists(results_dir):
        roads = parse_adafrenetic_json(results_dir)

    log_file = os.path.join(adafrenetic_dir, "adafrenetic_output.log")
    if not roads and os.path.exists(log_file):
        roads = parse_adafrenetic_log(log_file)

    if not roads:
        print("  No AdaFrenetic output found. Generating sample roads...")
        roads = generate_sample_roads(num_roads=20, map_size=map_size)
        print(f"  Generated {len(roads)} sample roads")

    return roads


# =============================================================================
# CommonRoad ST Vehicle Model Integration
# =============================================================================

class STVehicle:
    """
    Wraps CommonRoad's Single-Track dynamic vehicle model.

    State vector: [x, y, delta, v, psi, psi_dot, beta]
      - x, y: global position
      - delta: front wheel steering angle
      - v: longitudinal velocity
      - psi: yaw angle (heading)
      - psi_dot: yaw rate
      - beta: slip angle at vehicle center

    Input: [delta_dot, a_x]
      - delta_dot: steering angle velocity (rad/s)
      - a_x: longitudinal acceleration (m/s^2)
    """

    def __init__(self, x, y, heading, speed_ms):
        self.p = parameters_vehicle1()
        self.wheelbase = self.p.a + self.p.b
        self.max_steering = self.p.steering.max
        # State: [x, y, delta, v, psi, psi_dot, beta]
        self.state = np.array([x, y, 0.0, speed_ms, heading, 0.0, 0.0])

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def heading(self):
        return self.state[4]

    @property
    def speed(self):
        return self.state[3]

    @property
    def steering(self):
        return self.state[2]

    @property
    def slip_angle(self):
        return self.state[6]

    def step(self, desired_steering: float, target_speed_ms: float, dt: float):
        """Advance vehicle by dt seconds using the ST model."""
        # Clamp desired steering to physical limits
        desired_steering = np.clip(desired_steering, -self.max_steering, self.max_steering)

        # Convert desired steering angle to steering rate
        current_delta = self.state[2]
        delta_dot = (desired_steering - current_delta) / dt
        delta_dot = np.clip(delta_dot, -0.4, 0.4)  # physical steering rate limit

        # Simple P-controller for speed
        a_x = 2.0 * (target_speed_ms - self.state[3])

        u = [delta_dot, a_x]

        # Integrate using RK45
        def dynamics(t, x):
            return vehicle_dynamics_st(x, u, self.p)

        try:
            sol = solve_ivp(dynamics, [0, dt], self.state, method='RK45',
                            max_step=dt/2, rtol=1e-6, atol=1e-6)
            if sol.success:
                self.state = sol.y[:, -1]
            else:
                # Fallback: Euler step
                dx = vehicle_dynamics_st(self.state, u, self.p)
                self.state = self.state + np.array(dx) * dt
        except Exception:
            # Fallback: Euler step
            dx = vehicle_dynamics_st(self.state, u, self.p)
            self.state = self.state + np.array(dx) * dt

        # Keep velocity positive
        self.state[3] = max(self.state[3], 0.1)


# =============================================================================
# Pure Pursuit Controller
# =============================================================================

def pure_pursuit_steering(
    x: float, y: float, heading: float,
    road_points: np.ndarray,
    wheelbase: float,
    look_ahead_distance: float,
    max_steering: float
) -> float:
    """Calculate steering angle using Pure Pursuit algorithm."""
    car_pos = np.array([x, y])
    distances = np.linalg.norm(road_points - car_pos, axis=1)
    closest_idx = np.argmin(distances)

    # Find look-ahead point
    target_idx = closest_idx
    for i in range(closest_idx, len(road_points)):
        if np.linalg.norm(road_points[i] - car_pos) >= look_ahead_distance:
            target_idx = i
            break
    else:
        target_idx = len(road_points) - 1

    target = road_points[target_idx]

    dx = target[0] - x
    dy = target[1] - y
    angle_to_target = np.arctan2(dy, dx)
    alpha = angle_to_target - heading
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), look_ahead_distance)
    return np.clip(steering, -max_steering, max_steering)


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class SimulationResult:
    """Results from simulating one road."""
    is_valid: bool = True
    invalid_reason: str = ""
    oob_count: int = 0
    max_deviation: float = 0.0
    avg_deviation: float = 0.0
    min_oob_distance: float = 0.0
    reached_end: bool = False
    total_distance: float = 0.0
    # Stored for plotting the worst road
    road_points: np.ndarray = field(default_factory=lambda: np.array([]))
    centerline: np.ndarray = field(default_factory=lambda: np.array([]))
    trajectory: List[Tuple] = field(default_factory=list)
    oob_positions: List[Tuple] = field(default_factory=list)
    deviations: List[float] = field(default_factory=list)


def simulate_road(
    road_control_points: np.ndarray,
    speed_kmh: float = 70.0,
    oob_tolerance: float = 0.85,
    map_size: int = 200,
    dt: float = 0.02,
    max_sim_time: float = 60.0
) -> SimulationResult:
    """Run ST model simulation on a single road."""
    result = SimulationResult()

    # Validate
    is_valid, reason = validate_road(road_control_points, map_size)
    if not is_valid:
        result.is_valid = False
        result.invalid_reason = reason
        return result

    # Smooth road
    centerline = smooth_road(road_control_points, num_samples=500)

    # Initialize vehicle at road start
    start_dir = centerline[1] - centerline[0]
    start_heading = np.arctan2(start_dir[1], start_dir[0])
    speed_ms = speed_kmh / 3.6

    vehicle = STVehicle(
        x=centerline[0][0],
        y=centerline[0][1],
        heading=start_heading,
        speed_ms=speed_ms
    )

    oob_threshold = (ROAD_WIDTH / 2.0) * oob_tolerance
    centerline_line = LineString(centerline)

    # Simulation loop
    trajectory = []
    oob_positions = []
    deviations = []
    in_oob = False
    oob_events = 0
    reached_end = False

    for step in range(int(max_sim_time / dt)):
        car_pos = Point(vehicle.x, vehicle.y)
        trajectory.append((vehicle.x, vehicle.y))

        # Distance from road center
        deviation = car_pos.distance(centerline_line)
        deviations.append(deviation)

        # OOB check
        if deviation > oob_threshold:
            oob_positions.append((vehicle.x, vehicle.y))
            if not in_oob:
                oob_events += 1
                in_oob = True
        else:
            in_oob = False

        # Reached end?
        dist_to_end = np.linalg.norm(np.array([vehicle.x, vehicle.y]) - centerline[-1])
        if dist_to_end < 5.0 and step > 10:
            reached_end = True
            break

        # Gone way off-road?
        if deviation > ROAD_WIDTH * 3:
            break

        # Pure Pursuit steering
        look_ahead = max(5.0, vehicle.speed * 0.6)
        desired_steering = pure_pursuit_steering(
            vehicle.x, vehicle.y, vehicle.heading,
            centerline, vehicle.wheelbase,
            look_ahead, vehicle.max_steering
        )

        # Step vehicle
        vehicle.step(desired_steering, speed_ms, dt)

    # Compute metrics
    result.oob_count = oob_events
    result.max_deviation = max(deviations) if deviations else 0
    result.avg_deviation = float(np.mean(deviations)) if deviations else 0
    result.reached_end = reached_end

    half_road = ROAD_WIDTH / 2.0
    result.min_oob_distance = half_road - (max(deviations) if deviations else 0)

    if len(trajectory) > 1:
        traj = np.array(trajectory)
        result.total_distance = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    # Store data for plotting
    result.road_points = road_control_points
    result.centerline = centerline
    result.trajectory = trajectory
    result.oob_positions = oob_positions
    result.deviations = deviations

    return result


# =============================================================================
# CSV Output
# =============================================================================

CSV_COLUMNS = [
    'id rulare',
    'nr scenarii valide',
    'nr scenarii invalide',
    'iesiri de pe banda OBE',
    'eficienta/timp de generare (s)',
    'Viteza Setata (km/h)',
    'OBE Tolerance',
    'Dimensiune harta',
    'Valoare de fitness',
    'rezultat final',
]


def write_csv(csv_path: str, run_data: dict):
    """Write a single-row CSV for this run."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerow([run_data.get(col, '') for col in CSV_COLUMNS])


# =============================================================================
# Plot: Worst Road (lowest fitness)
# =============================================================================

def compute_road_boundaries(centerline: np.ndarray, width: float = ROAD_WIDTH):
    """Compute left/right road boundaries from centerline via perpendicular offsets."""
    left, right = [], []
    for i in range(len(centerline)):
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i == len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        normal = np.array([-tangent[1], tangent[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            normal = normal / norm_len
        half_w = width / 2.0
        left.append(centerline[i] + normal * half_w)
        right.append(centerline[i] - normal * half_w)
    return np.array(left), np.array(right)


def plot_worst_road(result: SimulationResult, speed_kmh: float, oob_tolerance: float,
                    save_path: str):
    """Plot the hardest road found by AdaFrenetic with the car trajectory."""
    centerline = result.centerline
    left_b, right_b = compute_road_boundaries(centerline)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Road surface
    road_x = np.concatenate([left_b[:, 0], right_b[::-1, 0]])
    road_y = np.concatenate([left_b[:, 1], right_b[::-1, 1]])
    ax.fill(road_x, road_y, color='#d9d9d9', alpha=0.7, label='Road surface')
    ax.plot(left_b[:, 0], left_b[:, 1], 'k-', linewidth=1.5, alpha=0.6)
    ax.plot(right_b[:, 0], right_b[:, 1], 'k-', linewidth=1.5, alpha=0.6)
    ax.plot(centerline[:, 0], centerline[:, 1], 'k--', linewidth=0.8, alpha=0.4,
            label='Centerline')

    # Control points from AdaFrenetic
    ax.plot(result.road_points[:, 0], result.road_points[:, 1],
            'bs', markersize=5, alpha=0.5, label='AdaFrenetic control points')

    # Car trajectory colored by deviation
    if len(result.trajectory) > 1:
        traj = np.array(result.trajectory)
        devs = np.array(result.deviations[:len(traj)])
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, ROAD_WIDTH / 2)
        lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm, linewidth=2.5)
        lc.set_array(devs[:-1])
        line = ax.add_collection(lc)
        plt.colorbar(line, ax=ax, label='Deviation from center (m)', shrink=0.6)

    # OOB positions
    if result.oob_positions:
        oob = np.array(result.oob_positions)
        ax.scatter(oob[:, 0], oob[:, 1], c='red', s=8, alpha=0.5,
                   zorder=5, label=f'OOB positions ({result.oob_count} events)')

    # Start/End markers
    if result.trajectory:
        ax.plot(*result.trajectory[0], 'g^', markersize=14, zorder=10, label='Start')
        ax.plot(*result.trajectory[-1], 'rv', markersize=14, zorder=10, label='End')

    status = "SUCCESS" if (result.reached_end and result.oob_count == 0) else "FAIL"
    metrics = (
        f"Result: {status}\n"
        f"OOB events: {result.oob_count}\n"
        f"min_oob_distance: {result.min_oob_distance:.2f} m\n"
        f"Max deviation: {result.max_deviation:.2f} m\n"
        f"Distance driven: {result.total_distance:.1f} m\n"
        f"Vehicle model: CommonRoad ST"
    )
    ax.text(0.02, 0.98, metrics, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Worst Road Found by AdaFrenetic (speed={speed_kmh}km/h, tol={oob_tolerance})',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Worst road plot saved: {save_path}")


# =============================================================================
# Plot: Evolution Progress (parsed from AdaFrenetic log)
# =============================================================================

def parse_evolution_data(log_file: str) -> List[dict]:
    """Parse test results from AdaFrenetic log in chronological order."""
    entries = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    i = 0
    test_idx = 0
    while i < len(lines):
        line = lines[i]

        # Match "test_outcome PASS/FAIL/INVALID"
        if 'test_outcome ' in line:
            outcome = line.strip().split('test_outcome ')[-1]
            entry = {'index': test_idx, 'outcome': outcome,
                     'min_oob_distance': None, 'accum_neg_oob': None, 'method': 'random'}
            test_idx += 1

            # Look at following lines for metrics
            for j in range(i + 1, min(i + 5, len(lines))):
                if 'Min oob_distance:' in lines[j]:
                    try:
                        entry['min_oob_distance'] = float(lines[j].strip().split('Min oob_distance: ')[-1])
                    except ValueError:
                        pass
                if 'Accumulated negative oob_distance:' in lines[j]:
                    try:
                        entry['accum_neg_oob'] = float(lines[j].strip().split('Accumulated negative oob_distance: ')[-1])
                    except ValueError:
                        pass

            # Look backwards for method info
            for j in range(max(0, i - 5), i):
                if 'Mutation function:' in lines[j]:
                    entry['method'] = lines[j].strip().split('Mutation function: ')[-1]
                elif 'Random generation.' in lines[j]:
                    entry['method'] = 'random'
                elif 'Entering crossover phase.' in lines[j]:
                    entry['method'] = 'crossover'

            entries.append(entry)
        i += 1

    return entries


def plot_evolution(entries: List[dict], speed_kmh: float, oob_tolerance: float,
                   save_path: str):
    """Plot how the genetic algorithm evolves over time.

    4 subplots:
      1. min_oob_distance per test (scatter) + running best (line)
      2. Cumulative FAIL count over tests
      3. Method used (color-coded scatter)
      4. accum_neg_oob per test
    """
    valid = [e for e in entries if e['min_oob_distance'] is not None]
    if len(valid) < 2:
        print("  Not enough valid tests to plot evolution.")
        return

    indices = [e['index'] for e in valid]
    oob_dists = [e['min_oob_distance'] for e in valid]
    accum_negs = [e['accum_neg_oob'] or 0 for e in valid]
    methods = [e['method'] for e in valid]

    # Running best (most negative min_oob_distance)
    running_best = []
    best_so_far = oob_dists[0]
    for d in oob_dists:
        best_so_far = min(best_so_far, d)
        running_best.append(best_so_far)

    # Cumulative FAILs (across ALL entries, not just valid)
    cum_fails = []
    fail_count = 0
    for e in entries:
        if e['outcome'] == 'FAIL':
            fail_count += 1
        cum_fails.append(fail_count)
    all_indices = [e['index'] for e in entries]

    # Method categories for coloring
    method_colors = {}
    method_labels = {}
    for m in methods:
        if m == 'random':
            method_colors[m] = '#1f77b4'
            method_labels[m] = 'Random'
        elif 'crossover' in m:
            method_colors[m] = '#2ca02c'
            method_labels[m] = 'Crossover'
        elif 'reverse' in m:
            method_colors[m] = '#d62728'
            method_labels[m] = 'Reverse'
        elif 'increase' in m:
            method_colors[m] = '#ff7f0e'
            method_labels[m] = 'Increase kappas'
        elif 'add' in m:
            method_colors[m] = '#9467bd'
            method_labels[m] = 'Add kappas'
        elif 'remove' in m:
            method_colors[m] = '#8c564b'
            method_labels[m] = 'Remove kappas'
        elif 'modify' in m:
            method_colors[m] = '#e377c2'
            method_labels[m] = 'Modify kappas'
        elif 'flip' in m:
            method_colors[m] = '#bcbd22'
            method_labels[m] = 'Flip sign'
        elif 'swap' in m:
            method_colors[m] = '#17becf'
            method_labels[m] = 'Split & swap'
        else:
            method_colors[m] = '#7f7f7f'
            method_labels[m] = m[:20]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # --- Subplot 1: min_oob_distance + running best ---
    ax1 = axes[0]
    ax1.scatter(indices, oob_dists, c='#1f77b4', s=15, alpha=0.4, label='Per test')
    ax1.plot(indices, running_best, 'r-', linewidth=2, label='Running best (most negative)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Road edge (oob_distance=0)')
    ax1.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.5, label='Mutation threshold (-0.5)')
    ax1.set_ylabel('min_oob_distance (m)')
    ax1.set_title(f'Evolution of Road Difficulty (speed={speed_kmh}km/h, tol={oob_tolerance})',
                  fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Cumulative FAILs ---
    ax2 = axes[1]
    ax2.plot(all_indices, cum_fails, 'r-', linewidth=2)
    ax2.fill_between(all_indices, cum_fails, alpha=0.15, color='red')
    ax2.set_ylabel('Cumulative FAILs')
    ax2.set_title('Failure Accumulation Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Method scatter (what strategy produced each test) ---
    ax3 = axes[2]
    plotted_labels = set()
    for idx, dist, m in zip(indices, oob_dists, methods):
        c = method_colors.get(m, '#7f7f7f')
        lbl = method_labels.get(m, m[:20])
        ax3.scatter(idx, dist, c=c, s=20, alpha=0.6,
                    label=lbl if lbl not in plotted_labels else None)
        plotted_labels.add(lbl)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('min_oob_distance (m)')
    ax3.set_title('Generation Method per Test', fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right', ncol=2)
    ax3.grid(True, alpha=0.3)

    # --- Subplot 4: accum_neg_oob ---
    ax4 = axes[3]
    ax4.bar(indices, accum_negs, width=1.0, color='#d62728', alpha=0.5)
    ax4.set_ylabel('accum_neg_oob')
    ax4.set_xlabel('Test Number (chronological)')
    ax4.set_title('Accumulated Negative OOB (severity of lane departure)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Evolution plot saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='CommonRoad Bridge Simulator for AdaFrenetic roads'
    )
    parser.add_argument('--adafrenetic-dir', type=str, default='../adafrenetic-sbst22',
                        help='Path to AdaFrenetic repo (default: ../adafrenetic-sbst22)')
    parser.add_argument('--speed', type=float, default=70.0,
                        help='Vehicle speed in km/h (default: 70)')
    parser.add_argument('--oob-tolerance', type=float, default=0.85,
                        help='OOB tolerance 0.0-1.0 (default: 0.85)')
    parser.add_argument('--map-size', type=int, default=500,
                        help='Map size in meters (default: 500)')
    parser.add_argument('--csv-file', type=str, default='',
                        help='Output CSV path (deprecated, use --output-dir)')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory for all results (CSV, plots, log)')
    return parser.parse_args()


def parse_log_stats(log_file: str) -> dict:
    """Extract all metrics from AdaFrenetic log without re-simulating."""
    stats = {
        'total': 0, 'pass': 0, 'fail': 0, 'invalid': 0, 'error': 0,
        'oob_distances': [], 'total_oob_events': 0,
        'elapsed': 0.0,
    }
    first_time = None
    last_time = None
    time_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    with open(log_file, 'r') as f:
        for line in f:
            # Track timestamps for elapsed time
            m = time_pattern.match(line)
            if m:
                try:
                    t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    if first_time is None:
                        first_time = t
                    last_time = t
                except ValueError:
                    pass

            if 'test_outcome ' in line:
                outcome = line.strip().split('test_outcome ')[-1]
                stats['total'] += 1
                if outcome == 'PASS':
                    stats['pass'] += 1
                elif outcome == 'FAIL':
                    stats['fail'] += 1
                elif outcome == 'INVALID':
                    stats['invalid'] += 1
                else:
                    stats['error'] += 1

            if 'Min oob_distance:' in line:
                try:
                    val = float(line.strip().split('Min oob_distance: ')[-1])
                    stats['oob_distances'].append(val)
                except ValueError:
                    pass

    if first_time and last_time:
        stats['elapsed'] = (last_time - first_time).total_seconds()

    # Count OOB events: number of roads with negative oob_distance
    stats['total_oob_events'] = sum(1 for d in stats['oob_distances'] if d < 0)

    return stats


def find_worst_road_from_log(log_file: str) -> np.ndarray:
    """Find the road with the lowest min_oob_distance by pairing road points with metrics from the log."""
    with open(log_file, 'r') as f:
        content = f.read()

    # Find all "Generated test using: [...]" blocks paired with their min_oob_distance
    # Pattern: road points logged, then later "Min oob_distance: X"
    road_pattern = r'Generated test using: \[(.*?)\]'
    road_matches = list(re.finditer(road_pattern, content, re.DOTALL))

    oob_pattern = r'Min oob_distance: ([\-\d.]+)'
    oob_matches = list(re.finditer(oob_pattern, content))

    if not oob_matches:
        return None

    # Match roads with their oob_distance by tracking which roads got simulated (not INVALID)
    # The oob_distances appear only for valid+simulated roads, in order
    valid_roads = []
    for road_match in road_matches:
        coord_pattern = r'(?:np\.float64\()?([\d.]+)\)?[,\s]+(?:np\.float64\()?([\d.]+)\)?'
        coords = re.findall(coord_pattern, road_match.group(1))
        if coords:
            points = np.array([(float(x), float(y)) for x, y in coords])
            # Check if next outcome after this road is not INVALID (search forward)
            pos = road_match.end()
            outcome_match = re.search(r'test_outcome (\w+)', content[pos:pos+2000])
            if outcome_match and outcome_match.group(1) in ('PASS', 'FAIL'):
                valid_roads.append(points)

    if not valid_roads or not oob_matches:
        return None

    # Pair valid roads with oob_distances (both in chronological order)
    n = min(len(valid_roads), len(oob_matches))
    best_idx = 0
    best_val = float('inf')
    for i in range(n):
        try:
            val = float(oob_matches[i].group(1))
            if val < best_val:
                best_val = val
                best_idx = i
        except ValueError:
            pass

    print(f"  Found worst road from log (fitness={best_val:.3f}, road #{best_idx + 1} of {n} valid)")
    return valid_roads[best_idx]


def find_worst_road_json(results_dir: str) -> np.ndarray:
    """Find the road with the lowest min_oob_distance from AdaFrenetic JSON results."""
    worst_road = None
    worst_fitness = float('inf')

    subdirs = sorted(glob.glob(os.path.join(results_dir, '*')), key=os.path.getmtime, reverse=True)
    for subdir in subdirs:
        json_files = sorted(glob.glob(os.path.join(subdir, 'test.*.json')))
        if not json_files:
            continue
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                if not data.get('is_valid', False):
                    continue
                points = np.array(data['road_points'])
                if len(points) < 3:
                    continue
                # We don't have min_oob_distance in JSON, collect all valid roads
                if worst_road is None:
                    worst_road = points
            except Exception:
                pass
        if worst_road is not None:
            break

    return worst_road


def main():
    args = parse_args()

    print("=" * 60)
    print("  AdaFrenetic Post-hoc Analysis")
    print("  (no re-simulation — stats from executor log)")
    print("=" * 60)

    # Determine output directory
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.output_dir:
        out_dir = args.output_dir
    elif args.csv_file:
        out_dir = os.path.dirname(args.csv_file) or 'results'
    else:
        out_dir = f"results/run_{run_id}_v{int(args.speed)}_t{args.oob_tolerance}"
    os.makedirs(out_dir, exist_ok=True)

    # Parse stats from AdaFrenetic log (already simulated by CommonRoad executor)
    log_file = os.path.join(args.adafrenetic_dir, "adafrenetic_output.log")
    if not os.path.exists(log_file):
        print(f"  ERROR: No log file found at {log_file}")
        sys.exit(1)

    print(f"\n  Parsing stats from log...")
    stats = parse_log_stats(log_file)
    valid_count = stats['pass'] + stats['fail']
    avg_fitness = float(np.mean(stats['oob_distances'])) if stats['oob_distances'] else 0.0
    min_fitness = min(stats['oob_distances']) if stats['oob_distances'] else 0.0

    print(f"\n  Speed: {args.speed} km/h | OOB Tolerance: {args.oob_tolerance}")
    print(f"  Total tests: {stats['total']}")
    print(f"  Valid: {valid_count} (PASS: {stats['pass']}, FAIL: {stats['fail']})")
    print(f"  Invalid: {stats['invalid']}")
    print(f"  OOB events: {stats['total_oob_events']}")
    print(f"  Best fitness (min_oob_distance): {min_fitness:.4f} m")
    print(f"  Avg fitness: {avg_fitness:.4f} m")
    print(f"  AdaFrenetic run time: {stats['elapsed']:.0f}s")

    # Write CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    if args.csv_file:
        csv_path = args.csv_file
    run_data = {
        'id rulare': run_id,
        'nr scenarii valide': valid_count,
        'nr scenarii invalide': stats['invalid'],
        'iesiri de pe banda OBE': stats['total_oob_events'],
        'eficienta/timp de generare (s)': f"{stats['elapsed']:.1f}",
        'Viteza Setata (km/h)': args.speed,
        'OBE Tolerance': args.oob_tolerance,
        'Dimensiune harta': args.map_size,
        'Valoare de fitness': f"{avg_fitness:.4f}",
        'rezultat final': f"{stats['pass']} PASS / {stats['fail']} FAIL",
    }
    write_csv(csv_path, run_data)
    print(f"\n  CSV saved: {csv_path}")

    # Plot worst road — find worst from log, simulate ONLY that one for the plot
    worst_road_points = find_worst_road_from_log(log_file)
    if worst_road_points is not None:
        print(f"\n  Simulating worst road for plot...")
        worst_result = simulate_road(
            worst_road_points, speed_kmh=args.speed,
            oob_tolerance=args.oob_tolerance,
            map_size=args.map_size
        )
        if worst_result.is_valid:
            print(f"  Worst road fitness: {worst_result.min_oob_distance:.2f}")
            plot_path = os.path.join(out_dir, "worst_road.png")
            plot_worst_road(worst_result, args.speed, args.oob_tolerance, plot_path)
        else:
            print(f"  Worst road invalid in post-hoc validation, skipping plot.")

    # Plot evolution progress from log
    entries = parse_evolution_data(log_file)
    if entries:
        evo_path = os.path.join(out_dir, "evolution.png")
        plot_evolution(entries, args.speed, args.oob_tolerance, evo_path)

    print(f"\n{'=' * 60}")
    print(f"  All results saved to: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
