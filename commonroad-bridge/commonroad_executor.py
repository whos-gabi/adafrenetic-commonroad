"""
CommonRoad Executor for AdaFrenetic
====================================
Replaces the mock/BeamNG executor with a CommonRoad Single-Track (ST) vehicle
model simulation. This gives AdaFrenetic real physics-based feedback for its
genetic algorithm, enabling guided evolution toward challenging roads.

The ST model includes tire slip, friction, and yaw dynamics — more realistic
than a kinematic bicycle model.
"""

import logging as log
import numpy as np
from scipy.integrate import solve_ivp
from shapely.geometry import LineString, Point

from code_pipeline.executors import AbstractTestExecutor
from self_driving.simulation_data import SimulationDataRecord
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st


ROAD_WIDTH = 8.0  # Standard SBST road width (meters)


# =============================================================================
# CommonRoad ST Vehicle Model
# =============================================================================

class STVehicle:
    """
    Wraps CommonRoad's Single-Track dynamic vehicle model.
    State: [x, y, delta, v, psi, psi_dot, beta]
    Input: [delta_dot, a_x]
    """

    def __init__(self, x, y, heading, speed_ms):
        self.p = parameters_vehicle1()
        self.wheelbase = self.p.a + self.p.b
        self.max_steering = self.p.steering.max
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

    def step(self, desired_steering, target_speed_ms, dt):
        desired_steering = np.clip(desired_steering, -self.max_steering, self.max_steering)
        delta_dot = np.clip((desired_steering - self.state[2]) / dt, -0.4, 0.4)
        a_x = 2.0 * (target_speed_ms - self.state[3])
        u = [delta_dot, a_x]

        try:
            sol = solve_ivp(
                lambda t, x: vehicle_dynamics_st(x, u, self.p),
                [0, dt], self.state, method='RK45',
                max_step=dt / 2, rtol=1e-6, atol=1e-6
            )
            if sol.success:
                self.state = sol.y[:, -1]
            else:
                dx = vehicle_dynamics_st(self.state, u, self.p)
                self.state = self.state + np.array(dx) * dt
        except Exception:
            dx = vehicle_dynamics_st(self.state, u, self.p)
            self.state = self.state + np.array(dx) * dt

        self.state[3] = max(self.state[3], 0.1)


# =============================================================================
# Pure Pursuit Controller
# =============================================================================

def pure_pursuit_steering(x, y, heading, road_points, wheelbase, look_ahead, max_steering):
    car_pos = np.array([x, y])
    distances = np.linalg.norm(road_points - car_pos, axis=1)
    closest_idx = np.argmin(distances)

    target_idx = closest_idx
    for i in range(closest_idx, len(road_points)):
        if np.linalg.norm(road_points[i] - car_pos) >= look_ahead:
            target_idx = i
            break
    else:
        target_idx = len(road_points) - 1

    target = road_points[target_idx]
    dx = target[0] - x
    dy = target[1] - y
    alpha = np.arctan2(dy, dx) - heading
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), look_ahead)
    return np.clip(steering, -max_steering, max_steering)


# =============================================================================
# CommonRoad Executor
# =============================================================================

class CommonRoadExecutor(AbstractTestExecutor):
    """
    Executor that simulates each road using CommonRoad's Single-Track vehicle
    model and returns real PASS/FAIL + OOB metrics to AdaFrenetic.
    """

    def __init__(self, result_folder, map_size,
                 time_budget=None, generation_budget=None, execution_budget=None,
                 oob_tolerance=0.85, max_speed_in_kmh=70,
                 road_visualizer=None):
        super().__init__(result_folder, map_size,
                         time_budget=time_budget,
                         generation_budget=generation_budget,
                         execution_budget=execution_budget,
                         road_visualizer=road_visualizer)
        self.oob_tolerance = oob_tolerance
        self.speed_ms = max_speed_in_kmh / 3.6
        self.speed_kmh = max_speed_in_kmh
        log.info(f"CommonRoad Executor: speed={max_speed_in_kmh}km/h, "
                 f"oob_tolerance={oob_tolerance}")

    def _execute(self, the_test):
        super()._execute(the_test)

        # Extract centerline from interpolated points (x, y, z, width)
        interp = the_test.interpolated_points
        if not interp or len(interp) < 4:
            return "ERROR", "Road too short for simulation", []

        centerline = np.array([(p[0], p[1]) for p in interp])
        centerline_line = LineString(centerline)

        # Initialize vehicle at road start
        start_dir = centerline[1] - centerline[0]
        start_heading = np.arctan2(start_dir[1], start_dir[0])

        vehicle = STVehicle(
            x=centerline[0][0],
            y=centerline[0][1],
            heading=start_heading,
            speed_ms=self.speed_ms
        )

        # Simulation parameters
        dt = 0.02  # 50 Hz
        max_sim_time = 30.0  # seconds
        car_half_width = 0.84  # ~1.68m vehicle width / 2
        half_road = ROAD_WIDTH / 2.0

        # Simulation loop
        execution_data = []
        oob_counter = 0
        in_oob = False
        max_oob_pct = 0.0

        for step in range(int(max_sim_time / dt)):
            sim_time = step * dt
            car_point = Point(vehicle.x, vehicle.y)

            # Distance from road center
            deviation = car_point.distance(centerline_line)

            # OOB metrics
            # oob_distance: positive = safe margin, negative = crossed edge
            oob_distance = half_road - deviation

            # oob_percentage: fraction of car outside road (approximation)
            edge_dist = half_road - deviation  # distance from car center to road edge
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
            max_oob_pct = max(max_oob_pct, oob_pct)

            # Build SimulationDataRecord
            record = SimulationDataRecord(
                timer=sim_time,
                pos=(vehicle.x, vehicle.y, 0.0),
                dir=(np.cos(vehicle.heading), np.sin(vehicle.heading), 0.0),
                vel=(vehicle.speed * np.cos(vehicle.heading),
                     vehicle.speed * np.sin(vehicle.heading), 0.0),
                steering=vehicle.steering,
                steering_input=0.0,
                brake=0.0,
                brake_input=0.0,
                throttle=0.5,
                throttle_input=0.5,
                wheelspeed=vehicle.speed,
                vel_kmh=int(vehicle.speed * 3.6),
                is_oob=is_oob,
                oob_counter=oob_counter,
                max_oob_percentage=max_oob_pct,
                oob_distance=oob_distance,
                oob_percentage=oob_pct
            )
            execution_data.append(record)

            # Check if car reached end of road
            dist_to_end = np.linalg.norm(
                np.array([vehicle.x, vehicle.y]) - centerline[-1]
            )
            if dist_to_end < 5.0 and step > 10:
                break

            # Car went way off-road — stop simulation
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
            vehicle.step(desired_steering, self.speed_ms, dt)

        # Determine outcome
        if oob_counter > 0:
            test_outcome = "FAIL"
            description = "Car drove out of the lane"
        else:
            test_outcome = "PASS"
            description = "Successful test"

        return test_outcome, description, execution_data

    def _close(self):
        pass
