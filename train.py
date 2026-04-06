"""
=====================================================================
  Training Script — RL Lunar Landing Rover
  Project: RL-Based Lunar Landing Rover
  Name   : Swaroop Bhowmik
  Reg No : 23BAI10722
=====================================================================

Usage:
    python train.py [--episodes N] [--save checkpoint]

The script trains the DQNAgent on LunarRoverEnv, logs per-episode
metrics, and saves the best-performing weights automatically.
"""

import argparse
import numpy as np
import time
import os

from lunar_lander_env import LunarRoverEnv
from dqn_agent import DQNAgent


# ================================================================
#   Training loop
# ================================================================

def train(num_episodes: int = 500, save_path: str = "best_model"):

    env   = LunarRoverEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim       = state_dim,
        action_dim      = action_dim,
        lr              = 1e-3,
        gamma           = 0.99,
        epsilon_start   = 1.0,
        epsilon_end     = 0.05,
        epsilon_decay   = 0.995,
        batch_size      = 64,
        buffer_capacity = 50_000,
        target_update_freq = 200,
    )

    # Metrics
    episode_rewards = []
    episode_lengths = []
    losses_per_ep   = []
    landings        = 0
    crashes         = 0
    best_avg_reward = -np.inf

    print("=" * 60)
    print("  RL Lunar Landing Rover — DQN Training")
    print(f"  Name   : Swaroop Bhowmik | Reg: 23BAI10722")
    print(f"  Episodes: {num_episodes} | State dim: {state_dim} | Actions: {action_dim}")
    print("=" * 60)

    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        ep_loss      = []
        ep_soft_land = False
        ep_crash     = False
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done   = terminated or truncated

            agent.store(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)

            state        = next_state
            total_reward += reward

            if info.get("soft_landed"):
                ep_soft_land = True
            if info.get("crashed"):
                ep_crash = True

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(info["steps"])
        losses_per_ep.append(np.mean(ep_loss) if ep_loss else 0.0)

        if ep_soft_land:
            landings += 1
        if ep_crash:
            crashes  += 1

        # Log every 20 episodes
        if ep % 20 == 0 or ep == 1:
            avg_r   = np.mean(episode_rewards[-20:])
            avg_len = np.mean(episode_lengths[-20:])
            avg_los = np.mean(losses_per_ep[-20:])
            land_rt = landings / ep * 100
            print(
                f"Ep {ep:4d}/{num_episodes} | "
                f"AvgR(20)={avg_r:+8.2f} | "
                f"AvgLen={avg_len:5.1f} | "
                f"AvgLoss={avg_los:.4f} | "
                f"Eps={agent.epsilon:.3f} | "
                f"LandRate={land_rt:.1f}%"
            )

            # Save best model
            if avg_r > best_avg_reward and ep >= 40:
                best_avg_reward = avg_r
                agent.save(save_path)
                print(f"  >> New best avg reward: {best_avg_reward:+.2f} — model saved.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Total episodes : {num_episodes}")
    print(f"  Soft landings  : {landings} ({landings/num_episodes*100:.1f}%)")
    print(f"  Crashes        : {crashes}  ({crashes/num_episodes*100:.1f}%)")
    print(f"  Best avg reward: {best_avg_reward:+.2f}")
    print(f"  Training time  : {elapsed:.1f}s")
    print("=" * 60)

    # Save final rewards for plotting
    np.save("rewards.npy", np.array(episode_rewards))
    print("  Episode rewards saved -> rewards.npy")

    env.close()
    return episode_rewards


# ================================================================
#   Entry point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Lunar Rover")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes (default: 500)")
    parser.add_argument("--save", type=str, default="best_model",
                        help="Checkpoint filename prefix (default: best_model)")
    args = parser.parse_args()

    train(num_episodes=args.episodes, save_path=args.save)
