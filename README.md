# RL-Based Lunar Landing Rover

**Name:** Swaroop Bhowmik  
**Registration Number:** 23BAI10722

---

## Project Overview

This project implements a Reinforcement Learning (RL) agent that learns to autonomously land a rover on the lunar surface using a custom Gymnasium environment and a from-scratch Deep Q-Network (DQN) algorithm implemented entirely in NumPy.

The agent observes the rover's position, velocity, angle, fuel level, and ground-contact state, and learns to fire thrusters in order to achieve a safe soft landing on the designated pad.

---

## Project Structure

```
lunar_lander_rl/
|
|-- lunar_lander_env.py    # Custom Gymnasium environment
|-- dqn_agent.py           # DQN agent + QNetwork + ReplayBuffer
|-- train.py               # Training loop with logging & checkpointing
|-- evaluate.py            # Evaluation & ASCII training-curve plotter
|-- utils.py               # Smoothing, logging, hyperparameter registry
|-- requirements.txt       # Python dependencies
|-- README.md              # This file
```

---

## Environment Details

### Observation Space (8-dimensional continuous)

| Index | Variable      | Description                    | Range          |
|-------|---------------|--------------------------------|----------------|
| 0     | x_pos         | Horizontal position            | [-1.5, 1.5]    |
| 1     | y_pos         | Vertical position              | [-0.5, 1.5]    |
| 2     | x_vel         | Horizontal velocity            | [-2.0, 2.0]    |
| 3     | y_vel         | Vertical velocity              | [-2.0, 2.0]    |
| 4     | angle         | Rover tilt (radians)           | [-pi, pi]      |
| 5     | ang_vel       | Angular velocity               | [-5.0, 5.0]    |
| 6     | fuel_norm     | Remaining fuel (normalised)    | [0.0, 1.0]     |
| 7     | leg_contact   | Ground contact flag            | {0, 1}         |

### Action Space (Discrete - 4 actions)

| Action | Description               |
|--------|---------------------------|
| 0      | Coast / Do nothing        |
| 1      | Fire main engine (thrust) |
| 2      | Fire left side thruster   |
| 3      | Fire right side thruster  |

### Reward Function

| Event                        | Reward         |
|------------------------------|----------------|
| Reduction in distance to pad | Positive shaping|
| Soft landing achieved        | +200           |
| Crash or out-of-bounds       | -100           |
| Time step (efficiency)       | -0.01 per step |
| Fuel consumption             | -0.3 per unit  |

---

## Algorithm: Deep Q-Network (DQN)

The agent uses a fully-connected Q-Network with the following features:

- **Architecture:** Input(8) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Output(4, Linear)
- **Optimiser:** Adam with learning rate 1e-3
- **Exploration:** Epsilon-greedy (ε: 1.0 → 0.05, decay 0.995)
- **Experience Replay:** Circular buffer of 50,000 transitions
- **Target Network:** Hard update every 200 steps (stabilises training)
- **Discount Factor (γ):** 0.99

**Bellman Update:**

```
Q(s,a) ← r + γ · max_a' Q_target(s', a')
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Setup

```bash
# Clone or copy the project folder
cd lunar_lander_rl

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the Agent

```bash
python train.py --episodes 500 --save best_model
```

Options:
- `--episodes` : Number of training episodes (default: 500)
- `--save`     : Filename prefix for checkpoint (default: best_model)

### Evaluate the Agent

```bash
python evaluate.py --model best_model.npz --episodes 20
```

Options:
- `--model`    : Path to saved model weights (.npz)
- `--episodes` : Number of evaluation episodes (default: 20)
- `--render`   : Print ASCII state each step
- `--plot`     : Display ASCII training reward curve

### Quick Demo (no saved model)

```bash
python evaluate.py --episodes 5 --render
```

---

## Key Results (Expected after 500 episodes)

| Metric               | Value (approx.) |
|----------------------|-----------------|
| Mean Episode Reward  | +150 to +220    |
| Soft Landing Rate    | 55% – 75%       |
| Convergence Episode  | ~300–400        |
| Fuel Efficiency      | 60% – 80%       |

---

## Module Reference

### `lunar_lander_env.py` — LunarRoverEnv
- `reset()` → returns initial observation
- `step(action)` → returns `(obs, reward, terminated, truncated, info)`
- `render()` → ASCII state printout
- `close()` → cleanup

### `dqn_agent.py` — DQNAgent
- `select_action(state, training)` → integer action
- `store(s, a, r, ns, done)` → push to replay buffer
- `learn()` → sample batch, compute loss, gradient step
- `decay_epsilon()` → anneal exploration
- `save(path)` / `load(path)` → checkpoint weights

### `utils.py` — Utilities
- `smooth(values, window)` → moving average
- `EpisodeLogger` → per-episode metrics accumulator
- `print_hyperparams()` → display default hyperparameters

---

## Physics Model

The rover dynamics use Euler integration over a fixed 20 ms time step:

- Moon gravity: **-1.622 m/s²** (1/6th of Earth)
- Main engine: 13.0 N thrust opposing angle direction
- Side thrusters: ±0.6 rad/s² angular impulse
- Landing criteria: ground contact + |vx|<0.15 + |vy|<0.15 + |angle|<0.2 rad

---

## Dependencies

| Package     | Version   | Purpose              |
|-------------|-----------|----------------------|
| numpy       | ≥ 1.24.0  | All numerical ops    |
| gymnasium   | ≥ 0.29.0  | Env base class / API |

No deep-learning framework (PyTorch/TensorFlow) is used. The neural network and Adam optimiser are implemented entirely from scratch in NumPy.

---

## License

This project is submitted for academic purposes.  
**Name:** Swaroop Bhowmik | **Reg:** 23BAI10722
