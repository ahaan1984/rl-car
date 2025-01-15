import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from collections import deque
import random

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        return self.fc5(x)

class Agent(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HYPERPARAMS
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_max = 0.96
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.update_target_frequency = 100

        self.policy_net = Network(state_size, action_size).to(self.device)
        self.target_net = Network(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimiser = optim.AdamW(self.policy_net.parameters())
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.Tensor(np.array(state, dtype=np.float32)).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # batch data
        states = torch.Tensor(np.array([transition[0] for transition in minibatch], dtype=np.float32)).to(self.device)
        actions = torch.LongTensor([transition[1] for transition in minibatch]).to(self.device)
        rewards = torch.Tensor(np.array([transition[2] for transition in minibatch], dtype=np.float32)).to(self.device)
        next_states = torch.Tensor(np.array([transition[3] for transition in minibatch], dtype=np.float32)).to(self.device)
        dones = torch.Tensor(np.array([transition[4] for transition in minibatch], dtype=np.float32)).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.HuberLoss()(current_q_values.squeeze(), target_q_values)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


