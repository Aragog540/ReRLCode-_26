"""
=====================================================================
  Deep Q-Network (DQN) Agent
  Project: RL-Based Lunar Landing Rover
  Name   : Swaroop Bhowmik
  Reg No : 23BAI10722
=====================================================================
"""

import numpy as np
import random
from collections import deque


# ============================================================
#   Neural Network (NumPy-only, no external DL framework)
# ============================================================

class DenseLayer:
    """Fully-connected layer with Xavier initialisation."""

    def __init__(self, in_dim, out_dim, activation="relu"):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, x):
        self._input = x
        z = x @ self.W + self.b
        if self.activation == "relu":
            self._z = z
            return np.maximum(0, z)
        return z  # linear

    def backward(self, grad_out):
        if self.activation == "relu":
            grad_out = grad_out * (self._z > 0)
        self.dW = self._input.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def update(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for (p, g, m, v) in [
            (self.W, self.dW, self.mW, self.vW),
            (self.b, self.db, self.mb, self.vb)
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class QNetwork:
    """3-hidden-layer Q-Network: state -> Q-values for each action."""

    def __init__(self, state_dim, action_dim, hidden=(128, 128, 64)):
        dims = [state_dim] + list(hidden) + [action_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            act = "relu" if i < len(dims) - 2 else "linear"
            self.layers.append(DenseLayer(dims[i], dims[i+1], act))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def copy_weights_from(self, other):
        for a, b in zip(self.layers, other.layers):
            a.W[:] = b.W
            a.b[:] = b.b


# ============================================================
#   Experience Replay Buffer
# ============================================================

class ReplayBuffer:
    """Circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int32),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
#   DQN Agent
# ============================================================

class DQNAgent:
    """
    Deep Q-Network agent with:
        - epsilon-greedy exploration
        - experience replay
        - target network (hard update)
        - Adam optimiser
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=50_000,
        target_update_freq=200,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lr         = lr
        self.gamma      = gamma
        self.epsilon    = epsilon_start
        self.eps_end    = epsilon_end
        self.eps_decay  = epsilon_decay
        self.batch_size = batch_size
        self.target_upd = target_update_freq

        self.q_net      = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.copy_weights_from(self.q_net)

        self.memory     = ReplayBuffer(buffer_capacity)
        self.steps_done = 0

    # -------------------------------------------------------
    def select_action(self, state, training=True):
        """Epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        q_vals = self.q_net.forward(state[np.newaxis, :])[0]
        return int(np.argmax(q_vals))

    # -------------------------------------------------------
    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # -------------------------------------------------------
    def learn(self):
        """Sample a mini-batch and perform one gradient step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Current Q-values
        q_all    = self.q_net.forward(states)        # (B, A)
        q_vals   = q_all[np.arange(self.batch_size), actions]

        # Target Q-values (Bellman equation)
        q_next   = self.target_net.forward(next_states)  # (B, A)
        q_target = rewards + self.gamma * q_next.max(axis=1) * (1 - dones)

        # MSE loss gradient w.r.t. network output
        loss     = np.mean((q_vals - q_target) ** 2)
        grad_out = np.zeros_like(q_all)
        grad_out[np.arange(self.batch_size), actions] = \
            2 * (q_vals - q_target) / self.batch_size

        self.q_net.backward(grad_out)
        self.q_net.update(self.lr)

        # Hard target update
        self.steps_done += 1
        if self.steps_done % self.target_upd == 0:
            self.target_net.copy_weights_from(self.q_net)

        return float(loss)

    # -------------------------------------------------------
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # -------------------------------------------------------
    def save(self, path):
        """Persist network weights to a .npz file."""
        params = {}
        for i, layer in enumerate(self.q_net.layers):
            params[f"W{i}"] = layer.W
            params[f"b{i}"] = layer.b
        np.savez(path, **params)
        print(f"[Agent] Weights saved -> {path}.npz")

    def load(self, path):
        """Load weights from a .npz file."""
        data = np.load(path)
        for i, layer in enumerate(self.q_net.layers):
            layer.W[:] = data[f"W{i}"]
            layer.b[:] = data[f"b{i}"]
        self.target_net.copy_weights_from(self.q_net)
        print(f"[Agent] Weights loaded <- {path}")
