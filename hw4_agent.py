import argparse
import logging
import timeit

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import gymnasium as gym

import hw4_utils as utils

torch.set_num_threads(4)

logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


class DQN(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 80, 85, 1
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size will be calculated below
        self.fc1 = nn.Linear(self.flat_size(), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)

    def flat_size(self):
        x = torch.zeros(1, self.iC, self.iH, self.iW)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()

    def forward(self, X):
        """
        X - bsz x iC x iH x iW observations (in order)
        returns:
          bsz x naction action logits
        """
        bsz = X.size()[0]

        Z = F.gelu(self.conv3(
              F.gelu(self.conv2(
                F.gelu(self.conv1(X))))))

        # flatten with MLP
        Z = F.gelu(self.fc1(Z.view(bsz, -1)))

        return self.fc2(Z)

    def get_action(self, x, eps=0.1):
        """
        x - 1 x iC x iH x iW
        returns:
        int index of action
        """
        if np.random.random() <= eps:
            return np.random.randint(0, self.fc2.out_features)
        else:
            with torch.no_grad():
                q_values = self(x.unsqueeze(0).float())
                action = q_values.argmax(-1).squeeze().item()
            return action


def train(args):
    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()

    model = DQN(naction, args)
    target_model = DQN(naction, args)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def lr_lambda(epoch):
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    memory = utils.ReplayBuffer(args.buffer_size, args.device)
    obs = env.reset()[0]
    total_reward = 0
    frame = 0

    def checkpoint():
        if args.save_path is None:
            return
        logging.info("Saving checkpoint to {}".format(args.save_path))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args}, args.save_path)

    timer = timeit.default_timer
    last_checkpoint_time = timer()

    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame

        state = utils.preprocess_observation(obs)
        action = model.get_action(state, args.epsilon)
        next_obs, reward, *_ = env.step(action)
        next_state = utils.preprocess_observation(next_obs)
        memory.add(state, action, reward, next_state)

        obs = next_obs
        total_reward += reward
        frame += 1

        if frame > args.learning_starts and frame % args.train_freq == 0:
            states, actions, rewards, next_states = memory.sample(args.batch_size)
            q_values = model(states)
            next_q_values = target_model(next_states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = next_q_values.max(1)[0]
            target = rewards + args.gamma * next_q_values

            loss = F.smooth_l1_loss(q_values, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if frame > args.learning_starts and frame % args.update_target_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if frame > args.learning_starts and frame % args.eval_freq == 0:
            test_env = gym.make(args.env)
            test_total_reward = 0
            test_obs = test_env.reset()
            test_state = utils.preprocess_observation(test_obs)
            test_action = model.get_action(test_state, 0.05)
            test_obs, test_reward, *_ = test_env.step(test_action)
            test_total_reward += test_reward
            test_env.close()

            if frame % 640 == 0:
                logging.info(f'Frame: {frame}, Test reward: {test_total_reward}')

        if frame > args.learning_starts and frame % args.lr_decay_step == 0:
            scheduler.step()

        if frame % 640 == 0:
            end_time = timer()
            fps = (frame - start_frame) / (end_time - start_time)
            logging.info(f'Frame: {frame}, FPS: {fps}, Total reward: {total_reward}')

        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()

        if frame > 0 and frame % (args.eval_every * args.batch_size) == 0:
            utils.validate(model, render=args.render)
            model.train()

    env.close()





parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--unroll_length", default=80, type=int, 
                    help="unroll length (time dimension)")
parser.add_argument("--total_frames", default=1000000, type=int, 
                    help="total environment frames to train for")
parser.add_argument("--buffer_size", default=1000000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--epsilon", default=0.1, type=float)
parser.add_argument("--learning_rate", default=0.00025, type=float)
parser.add_argument("--total_steps", default=1000000, type=int)
parser.add_argument("--learning_starts", default=50000, type=int)
parser.add_argument("--train_freq", default=4, type=int)
parser.add_argument("--update_target_freq", default=10000, type=int)
parser.add_argument("--eval_freq", default=10000, type=int)
parser.add_argument("--lr_decay_step", default=100000, type=int)
parser.add_argument("--hidden_dim", default=512, type=int)
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")
parser.add_argument("--mode", default="train", choices=["train", "valid"], 
                    help="training or validation mode")

if __name__ == "__main__":
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    
    if args.mode == "train":
        train(args)
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env

        model = DQN(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(args.device)
        args = saved_args

        utils.validate(model, args)

