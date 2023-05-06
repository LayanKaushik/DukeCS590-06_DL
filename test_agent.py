import argparse
import logging
import timeit

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

from torchvision.transforms import Resize, ToPILImage, ToTensor

import gymnasium as gym

import test_utils as utils
from test_QNet import DeepQNetwork
from test_replay import ReplayBuffer

torch.set_num_threads(4)


logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

class DQNAgent:
    def __init__(self, nactions, arg):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_target_frequency = update_target_frequency
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q = self.build_model().to(self.device)
        self.Q_target = self.build_model().to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        
        self.memory = deque(maxlen=self.buffer_size)
        self.steps = 0

    def build_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space.n)
        )
        return model

    def choose_action(self, state):
        if random.random() > self.epsilon:
            return self.choose_greedy_action(state)
        else:
            return self.env.action_space.sample()

    def choose_greedy_action(self, state):
        state_tensor = preprocess_observation(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.Q(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        target_q_values = self.target_model.predict(next_states)
        target_q_values[dones] = 0

        q_values = self.model.predict(states)
        for i, action in enumerate(actions):
            q_values[i, action] = rewards[i] + self.gamma * np.max(target_q_values[i])

        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([preprocess_observation(s) for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack([preprocess_observation(ns) for ns in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q_values = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
                        next_q_values = self.Q_target(next_states).max(1).values
        target_q_values = torch.where(dones, rewards, rewards + self.gamma * next_q_values)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        if self.steps % self.update_target_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

def train(agent, env, episodes=1000, render=False):
    total_reward = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render()

            action = agent.choose_action(state)
            result = env.step(action)  # Modify this line
            next_state, reward, done = result[:3]  # Add this line

            episode_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

        total_reward += episode_reward
        print(f"Episode {episode + 1} - Reward: {episode_reward}")

    return total_reward


