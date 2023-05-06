import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def preprocess(observation):
    # Preprocess observation here
    pass

class DQNAgent:
    def __init__(self, env, buffer_size, batch_size, learning_rate, gamma, target_update_freq, epsilon_start, epsilon_end, epsilon_decay):
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_q_net = QNetwork(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.q_net(state).max(1)[1].item()
            
    def choose_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state).max(1)[1].item()


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            target_q_values = self.target_q_net(next_states)

            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            target_q = target_q_values.gather(1, next_actions)

            target_values = rewards + (1 - dones) * self.gamma * target_q

        loss = nn.MSELoss()(q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

def train(agent, num_episodes, render=False):
    for episode in range(num_episodes):
        state = preprocess(agent.env.reset())
        episode_reward = 0
        done = False

        while not done:
            if render:
                agent.env.render()

            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            next_state = preprocess(next_state)

            agent.buffer.store(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            episode_reward += reward

        agent.update_epsilon()
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        print("Episode: {}, Reward: {}, Epsilon: {:.4f}".format(episode, episode_reward, agent.epsilon))

    agent.env.close()


if __name__ == "__main__":
    env = gym.make("MsPacman-v0")

    # Hyperparameters
    buffer_size = 10000
    batch_size = 64
    learning_rate = 0.00025
    gamma = 0.99
    target_update_freq = 1000
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    num_episodes = 1000

    agent = DQNAgent(env, buffer_size, batch_size, learning_rate, gamma, target_update_freq, epsilon_start, epsilon_end, epsilon_decay)
    train(agent, num_episodes)

def validate(agent, render=False, nepisodes=1):
    torch.manual_seed(590060)
    np.random.seed(590060)

    total_reward = 0
    steps_alive = []
    for i in range(nepisodes):
        env = gym.make("MsPacman-v0")
        state = env.reset(seed=590060+i)
        prev_state = None
        step = 0
        # play until the agent dies or we exceed 50000 observations
        while env.ale.lives() == 3 and step < 50000:
            action = agent.choose_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(0.1)
            state = next_state
            step += 1
        steps_alive.append(step)

    print("Steps taken over each of {:d} episodes: {}".format(nepisodes, ", ".join(str(step) for step in steps_alive)))
    print("Total return after {:d} episodes: {:.3f}".format(nepisodes, total_reward))


# Train the agent
agent = DQNAgent(env)
train(agent, num_episodes=1000)

# Validate the agent's performance
validate(agent, render=True, nepisodes=5)
