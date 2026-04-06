"""
=====================================================================
  Utilities — RL Lunar Landing Rover
  Project: RL-Based Lunar Landing Rover
  Name   : Swaroop Bhowmik
  Reg No : 23BAI10722
=====================================================================
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Moving-average smoother
# ------------------------------------------------------------------ #

def smooth(values, window=20):
    """Return exponential moving average of a list of values."""
    if len(values) < 2:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ------------------------------------------------------------------ #
#  Replay-buffer statistics
# ------------------------------------------------------------------ #

def buffer_stats(replay_buffer):
    """Print basic statistics about the replay buffer contents."""
    if len(replay_buffer) == 0:
        print("[Buffer] Empty.")
        return
    states, actions, rewards, next_states, dones = \
        replay_buffer.sample(min(1000, len(replay_buffer)))
    print(f"[Buffer] Size: {len(replay_buffer)}")
    print(f"  Reward  mean/std : {rewards.mean():.3f} / {rewards.std():.3f}")
    print(f"  Action  counts   : { {i: int((actions==i).sum()) for i in range(4)} }")
    print(f"  Done    fraction : {dones.mean():.3f}")


# ------------------------------------------------------------------ #
#  Episode logger
# ------------------------------------------------------------------ #

class EpisodeLogger:
    """
    Accumulates per-episode metrics and provides formatted summaries.

    Usage:
        logger = EpisodeLogger()
        logger.log(episode=1, reward=-120.5, length=300, landed=False)
        logger.print_summary()
    """

    def __init__(self):
        self.episodes = []
        self.rewards  = []
        self.lengths  = []
        self.landings = 0
        self.crashes  = 0

    def log(self, episode, reward, length, landed=False, crashed=False):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.lengths.append(length)
        if landed:
            self.landings += 1
        if crashed:
            self.crashes  += 1

    def print_summary(self, last_n=50):
        n = min(last_n, len(self.rewards))
        recent = self.rewards[-n:]
        print(f"\n--- Logger Summary (last {n} episodes) ---")
        print(f"  Mean reward : {np.mean(recent):+.2f}")
        print(f"  Max  reward : {np.max(recent):+.2f}")
        print(f"  Landing rate: {self.landings/max(1,len(self.episodes))*100:.1f}%")
        print(f"  Crash   rate: {self.crashes /max(1,len(self.episodes))*100:.1f}%")


# ------------------------------------------------------------------ #
#  Hyperparameter registry
# ------------------------------------------------------------------ #

DEFAULT_HYPERPARAMS = {
    "lr"               : 1e-3,
    "gamma"            : 0.99,
    "epsilon_start"    : 1.0,
    "epsilon_end"      : 0.05,
    "epsilon_decay"    : 0.995,
    "batch_size"       : 64,
    "buffer_capacity"  : 50_000,
    "target_update_freq": 200,
    "hidden_layers"    : (128, 128, 64),
    "num_episodes"     : 500,
}


def print_hyperparams(params=None):
    if params is None:
        params = DEFAULT_HYPERPARAMS
    print("\n--- Hyperparameters ---")
    for k, v in params.items():
        print(f"  {k:<22}: {v}")
    print()
