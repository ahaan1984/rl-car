import random
from collections import deque

import numpy as np
import torch
from torch import nn, optim


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, output_size)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return self.fc3(x)

class Agent(nn.Module):
    def __init__(self, state_size:int, action_size:int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_max = 0.96
        self.epsilon_decay = 0.995
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.update_target_frequency = 100

        self.recent_rewards = deque(maxlen=100)
        self.baseline_reward = None
        self.performance_threshold = 0.8

        self.policy_net = Network(state_size, action_size).to(self.device)
        self.target_net = Network(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimiser = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps = 0

    def remember(self, state, action, reward, next_state, done) -> deque:
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)

        if self.baseline_reward is None and len(self.recent_rewards) >= 100:
            self.baseline_reward = sum(self.recent_rewards) / len(self.recent_rewards)

    def act(self, state):
        if np.random.default_rng().random() < self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            self.policy_net.eval()
            state = torch.Tensor(np.array(state, dtype=np.float32)).unsqueeze(0)
            q_values = self.policy_net(state)
            self.policy_net.train()
            return q_values.argmax().item()

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        self.policy_net.train()

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch, strict=True)

        states = torch.Tensor(np.array([transition[0] for transition in minibatch],
                                dtype=np.float32)).to(self.device)
        actions = torch.LongTensor([transition[1] for transition in minibatch]).to(self.device)
        rewards = torch.Tensor(np.array([transition[2] for transition in minibatch],
                                dtype=np.float32)).to(self.device)
        next_states = torch.Tensor(np.array([transition[3] for transition in minibatch],
                                dtype=np.float32)).to(self.device)
        dones = torch.Tensor(np.array([transition[4] for transition in minibatch],
                                dtype=np.float32)).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_q_values.unsqueeze(1))

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if len(self.recent_rewards) >= 100:
            current_avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)

            if self.baseline_reward is None:
                self.baseline_reward = current_avg_reward
            else:
                self.baseline_reward = 0.9 * self.baseline_reward + 0.1 * current_avg_reward

            performance_ratio = current_avg_reward / (self.baseline_reward + 1e-10)

            if performance_ratio < 0.8:
                self.epsilon = max(self.epsilon_max,
                                 self.epsilon / self.epsilon_decay)
            else:
                self.epsilon = min(self.epsilon_min,
                                 self.epsilon * self.epsilon_decay)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def update_target_network(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename:str) -> None:
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "epsilon": self.epsilon},
            filename)

    def load(self, filename:str) -> None:
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.epsilon = checkpoint["epsilon"]


