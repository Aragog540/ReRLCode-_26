"""
=====================================================================
  Lunar Landing Rover - Custom Gymnasium Environment
  Project: RL-Based Lunar Landing Rover
  Name   : Swaroop Bhowmik
  Reg No : 23BAI10722
=====================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LunarRoverEnv(gym.Env):
    """
    Custom Gymnasium environment simulating a Lunar Landing Rover.

    The rover must navigate a lunar surface after descent, reach the
    designated landing zone, and come to a safe stop while managing fuel.

    Observation Space (8 values):
        [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, fuel, leg_contact]

    Action Space (Discrete - 4):
        0 - Do nothing / coast
        1 - Fire main engine (thrust up)
        2 - Fire left thruster
        3 - Fire right thruster

    Reward Structure:
        +  Distance reduction to pad
        -  Fuel consumption penalty
        +  200 bonus for safe landing
        -  100 penalty for crash
        -  10  penalty per step (time efficiency)
        -  Out-of-bounds penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, gravity=-1.622, fuel_limit=200):
        super().__init__()
        self.render_mode = render_mode
        self.gravity = gravity          # Moon gravity m/s^2
        self.fuel_limit = fuel_limit    # Total fuel units

        # --- State bounds ---
        # [x, y, vx, vy, angle, ang_vel, fuel, leg_contact]
        low = np.array([-1.5, -0.5, -2.0, -2.0, -np.pi, -5.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([ 1.5,  1.5,  2.0,  2.0,  np.pi,  5.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # --- Physical constants ---
        self.dt = 0.02               # time step (seconds)
        self.MAIN_ENGINE_POWER = 13.0
        self.SIDE_ENGINE_POWER = 0.6
        self.MASS = 1.0
        self.PAD_X = 0.0
        self.PAD_Y = 0.0

        # Internal state
        self._state = None
        self._fuel_remaining = fuel_limit
        self._steps = 0
        self._prev_shaping = None

        # For rendering
        self.screen = None
        self.clock = None

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial position above the surface
        x_init = self.np_random.uniform(-0.8, 0.8)
        y_init = self.np_random.uniform(0.8, 1.4)
        vx_init = self.np_random.uniform(-0.3, 0.3)
        vy_init = self.np_random.uniform(-0.4, 0.0)
        angle_init = self.np_random.uniform(-0.3, 0.3)
        ang_vel_init = self.np_random.uniform(-0.1, 0.1)

        self._fuel_remaining = self.fuel_limit
        self._steps = 0
        self._prev_shaping = None

        self._state = np.array([
            x_init, y_init, vx_init, vy_init,
            angle_init, ang_vel_init,
            self._fuel_remaining / self.fuel_limit,
            0.0   # leg contact
        ], dtype=np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        x, y, vx, vy, angle, ang_vel, fuel_norm, _ = self._state
        self._steps += 1

        # --- Engine forces ---
        fuel_cost = 0.0
        ax, ay = 0.0, self.gravity

        if action == 1 and self._fuel_remaining > 0:        # Main engine
            thrust = self.MAIN_ENGINE_POWER / self.MASS
            ax += -np.sin(angle) * thrust
            ay +=  np.cos(angle) * thrust
            fuel_cost = 0.3
        elif action == 2 and self._fuel_remaining > 0:      # Left thruster
            ang_vel -= self.SIDE_ENGINE_POWER
            fuel_cost = 0.03
        elif action == 3 and self._fuel_remaining > 0:      # Right thruster
            ang_vel += self.SIDE_ENGINE_POWER
            fuel_cost = 0.03

        # Update kinematics (Euler integration)
        vx += ax * self.dt
        vy += ay * self.dt
        x  += vx * self.dt
        y  += vy * self.dt
        angle += ang_vel * self.dt
        ang_vel *= 0.99  # angular damping

        # Update fuel
        self._fuel_remaining = max(0, self._fuel_remaining - fuel_cost)

        # --- Leg contact detection ---
        leg_contact = 1.0 if y <= 0.05 else 0.0

        self._state = np.array([
            x, y, vx, vy, angle, ang_vel,
            self._fuel_remaining / self.fuel_limit,
            leg_contact
        ], dtype=np.float32)

        # --- Termination conditions ---
        terminated = False
        truncated  = False

        out_of_bounds = abs(x) > 1.5 or y > 2.0
        crashed = y < -0.05  # below ground without soft landing
        soft_landed = (
            leg_contact > 0.5 and
            abs(vx) < 0.15 and
            abs(vy) < 0.15 and
            abs(angle) < 0.2
        )

        if out_of_bounds or crashed or soft_landed:
            terminated = True
        if self._steps >= 1000:
            truncated = True

        # --- Reward shaping ---
        shaping = (
            -100 * np.sqrt(x**2 + y**2)          # distance to pad
            - 100 * np.sqrt(vx**2 + vy**2)        # velocity magnitude
            - 100 * abs(angle)                     # tilt penalty
            + 10  * leg_contact                    # reward for contact
        )
        reward = shaping - (self._prev_shaping if self._prev_shaping is not None else shaping)
        self._prev_shaping = shaping

        reward -= 0.3 * fuel_cost     # fuel penalty
        reward -= 0.01               # time penalty per step

        if soft_landed:
            reward += 200
        if crashed or out_of_bounds:
            reward -= 100

        obs  = self._get_obs()
        info = self._get_info()
        info["soft_landed"] = soft_landed
        info["crashed"]     = crashed or out_of_bounds

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.clip(self._state.copy(),
                       self.observation_space.low,
                       self.observation_space.high).astype(np.float32)

    def _get_info(self):
        x, y, vx, vy, angle, ang_vel, fuel_norm, leg = self._state
        return {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "angle": angle, "fuel_remaining": self._fuel_remaining,
            "steps": self._steps
        }

    # ------------------------------------------------------------------
    def render(self):
        """Simple ASCII render for headless environments."""
        x, y, vx, vy, angle, _, fuel_norm, leg = self._state
        print(f"[Step {self._steps:4d}] "
              f"pos=({x:+.3f},{y:+.3f}) "
              f"vel=({vx:+.3f},{vy:+.3f}) "
              f"angle={np.degrees(angle):+.1f}deg "
              f"fuel={fuel_norm*100:.1f}% "
              f"contact={'YES' if leg > 0.5 else 'NO '}")

    def close(self):
        if self.screen is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
