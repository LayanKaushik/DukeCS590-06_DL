import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque

# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

# Define the Deep Q-Learning agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon, gamma, learning_rate, buffer_size, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax(1).item()

    def update_model(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.update_model()

            print("Episode: {}, Total Reward: {}".format(episode, total_reward))

# Define the main function for running the training
def main():
    # Create the Ms. Pacman environment
    env = gym.make('MsPacman-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Define hyperparameters
    epsilon = 0.1
    gamma = 0.99
    learning_rate = 0.001
    buffer_size = 10000
    batch_size = 32
    episodes = 1000

    # Create the DQN agent
    agent = DQNAgent(state_dim, action_dim, epsilon, gamma, learning_rate, buffer_size, batch_size)

    # Train the agent
    agent.train(env, episodes)

if __name__ == '__main__':
    main()