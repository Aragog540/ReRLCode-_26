"""
=====================================================================
  Evaluation Script — RL Lunar Landing Rover
  Project: RL-Based Lunar Landing Rover
  Name   : Swaroop Bhowmik
  Reg No : 23BAI10722
=====================================================================

Usage:
    python evaluate.py [--model best_model.npz] [--episodes 20] [--render]
"""

import argparse
import numpy as np
import os

from lunar_lander_env import LunarRoverEnv
from dqn_agent import DQNAgent


def evaluate(model_path: str, num_episodes: int = 20, render: bool = False):
    env = LunarRoverEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"[Evaluate] WARNING: Model '{model_path}' not found. Using random weights.")

    agent.epsilon = 0.0  # fully greedy

    total_rewards = []
    landings, crashes = 0, 0

    print("\n" + "=" * 60)
    print("  Evaluation — RL Lunar Landing Rover")
    print(f"  Model: {model_path}")
    print(f"  Episodes: {num_episodes}")
    print("=" * 60)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        soft_landed = False
        crashed     = False

        while not done:
            if render:
                env.render()
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if info.get("soft_landed"):
                soft_landed = True
            if info.get("crashed"):
                crashed = True

        total_rewards.append(total_reward)
        result = "LANDED " if soft_landed else ("CRASHED" if crashed else "TIMEOUT")
        if soft_landed:
            landings += 1
        if crashed:
            crashes  += 1

        print(f"  Ep {ep:3d}: Reward={total_reward:+8.2f} | Steps={info['steps']:4d} "
              f"| Fuel={info['fuel_remaining']:5.1f} | {result}")

    print("\n--- Summary ---")
    print(f"  Mean Reward   : {np.mean(total_rewards):+.2f}")
    print(f"  Std  Reward   : {np.std(total_rewards):.2f}")
    print(f"  Min / Max     : {np.min(total_rewards):+.2f} / {np.max(total_rewards):+.2f}")
    print(f"  Soft Landings : {landings}/{num_episodes} ({landings/num_episodes*100:.1f}%)")
    print(f"  Crashes       : {crashes}/{num_episodes} ({crashes/num_episodes*100:.1f}%)")
    print("=" * 60)

    env.close()
    return total_rewards


def plot_training_curve(rewards_path: str = "rewards.npy"):
    """Print an ASCII training curve from saved rewards."""
    if not os.path.exists(rewards_path):
        print(f"[Plot] File '{rewards_path}' not found. Run train.py first.")
        return

    rewards = np.load(rewards_path)
    window  = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

    print("\n--- Training Curve (smoothed) ---")
    max_r = smoothed.max()
    min_r = smoothed.min()
    rows  = 15
    for r in range(rows, -1, -1):
        threshold = min_r + (max_r - min_r) * r / rows
        line = ""
        step = max(1, len(smoothed) // 60)
        for val in smoothed[::step]:
            line += "*" if val >= threshold else " "
        print(f"  {threshold:+7.1f} | {line}")
    print("          " + "-" * (len(smoothed[::step]) + 2))
    print(f"           Episodes 1 → {len(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Lunar Rover")
    parser.add_argument("--model",    type=str, default="best_model.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render",   action="store_true", help="ASCII render each step")
    parser.add_argument("--plot",     action="store_true", help="Print training curve")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)
    if args.plot:
        plot_training_curve()
