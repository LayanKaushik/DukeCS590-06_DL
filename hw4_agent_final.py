import argparse
import logging
import timeit

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

import gymnasium as gym

import hw4_utils_final as utils


torch.set_num_threads(4)


logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 85, 80, 1
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Update the flattened size based on the new architecture
        self.fc1 = nn.Linear(256, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)

    def forward(self, X, prev_state=None):
        bsz, T = X.size()[:2]

        Z = F.relu(self.bn1(self.conv1(X.view(-1, self.iC, self.iH, self.iW))))
        Z = self.pool2(F.relu(self.bn2(self.conv2(Z))))
        Z = F.relu(self.bn3(self.conv3(Z)))

        Z = F.relu(self.fc1(Z.view(bsz*T, -1))) # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)
        
        return self.fc2(Z), prev_state

    def get_action(self, x, prev_state):
        logits, prev_state = self(x, prev_state)
        action = logits.argmax(-1).squeeze().item()
        return action, prev_state
    
    
class Actor_Crtic(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 85, 80, 1
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        # the flattened size is 2688 assuming dims and convs above
        self.fc1 = nn.Linear(2688, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)
        self.fc3 = nn.Linear(naction, 1)        
        
    def forward(self, X, prev_state=None):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action logits, prev_state
        """
        bsz, T = X.size()[:2]

        Z = F.relu(self.bn3(self.conv3(
              F.relu(self.bn2(self.conv2(
                F.relu(self.bn1(self.conv1(X.view(-1, self.iC, self.iH, self.iW))))))))))

        # flatten with MLP
        Z = F.relu(self.fc1(Z.view(bsz*T, -1))) # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)
        
        Z1 = F.relu(self.fc2(Z))
        Z1 = Z1.view(bsz, T, -1)  
        Z2 = self.fc3(Z1)
        
        return Z2, prev_state
    
    def get_action(self, x, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        action, prev_state = self(x, prev_state)

        return action, prev_state


def advantage_step(stepidx, model_actor, model_critic, optimizer, scheduler, envs, observations, prev_state, gamma, entropy_coef, bsz=4):
    
    # def get_diff(observations):
    #     diff_observations = observations[:, :-1] - observations[:, 1:]
    #     return diff_observations
    
    if envs is None:
        envs = [gym.make(args.env) for _ in range(bsz)]
        observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
        observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
            [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)
        prev_state = None

    logits, rewards, actions, values = [], [], [], []
    not_terminated = torch.ones(bsz) # agent is still alive
    
    for t in range(args.unroll_length):
        
        logits_t, prev_state = model_actor(observations, prev_state) # logits are bsz x 1 x naction
        logits.append(logits_t)
        
        # if we lose a life, zero out all subsequent rewards
        still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
        not_terminated.mul_(still_alive.float())
        
        # sample actions
        actions_t = Categorical(logits=logits_t.squeeze(1)).sample()
        actions.append(actions_t.view(-1, 1)) # bsz x 1
        # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
        env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
        
        # if we lose a life, zero out all subsequent rewards
        still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
        not_terminated.mul_(still_alive.float())

        rewards_t = torch.tensor([eo[1] for eo in env_outputs])
        
        neg = -1 #adding negative reward upon death
        rewards.append(rewards_t * not_terminated + (1 - not_terminated) * neg)

        values_t, prev_state = model_critic(observations, prev_state)
        values.append(values_t)
        
        env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
        
        observations = torch.stack([utils.preprocess_observation(eo[0]) for eo in env_outputs]).unsqueeze(1)

        #need extra q-value for last critic value assessment
        if (t == args.unroll_length -1):
            values_t, prev_state = model_critic(observations, prev_state)
            values.append(values_t)

    values = torch.cat(values, dim=1)
    values = values.squeeze()

    adv = torch.zeros(bsz, args.unroll_length)
    reward_present = 0
    
    values = values[:, :-1]

    for r in range(args.unroll_length - 1, -1, -1):
        if r < args.unroll_length - 1:
            reward_present = rewards[r] + gamma * values[:, r + 1]
        else:
            reward_present = rewards[r]
        adv[:, r].copy_(reward_present)

    advantage = (adv - values).detach()

    critic_loss = F.mse_loss(adv, values)
    
    logits = torch.cat(logits, dim=1) # bsz x T x naction
    actions = torch.cat(actions, dim=1) # bsz x T 
    cross_entropy = F.cross_entropy(
        logits.view(-1, logits.size(2)), actions.view(-1), reduction='none')

    
    actor_loss = (cross_entropy.view_as(actions) * advantage).mean()

    # Entropy regularization
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    total_loss = actor_loss + critic_loss - entropy_coef * entropy

    # total_loss = actor_loss + critic_loss

    stats = {"mean_return": sum(r.mean() for r in rewards)/len(rewards),
             "actor_loss": actor_loss.item()}
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model_actor.parameters(), args.grad_norm_clipping)
    optimizer.step()
    scheduler.step()
    prev_state = observations
    # reset any environments that have ended
    for b in range(bsz):
        if not_terminated[b].item() == 0:
            obs = envs[b].reset(seed=stepidx+b)[0]
            observations[b].copy_(utils.preprocess_observation(obs))

    return stats, envs, observations, prev_state


def train(args):
    T = args.unroll_length
    B = args.batch_size
    gamma = args.gamma 
    entropy_coef = args.entropy_coef
    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()
    del env

    model = Actor(naction, args)
    model_prev = Actor_Crtic(naction, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    def lr_lambda(epoch): # multiplies learning rate by value returned; can be used to decay lr
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
    envs, observations, prev_state = None, None, None
    frame = 0
    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame
        stats, envs, observations, prev_state = advantage_step(
            frame, model, model_prev, optimizer, scheduler, envs, observations, prev_state, gamma, entropy_coef, bsz=B)
        frame += T*B # here steps means number of observations
        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()

        sps = (frame - start_frame) / (timer() - start_time)
        logging.info("Frame {:d} @ {:.1f} FPS: actor_loss {:.3f} | mean_ret {:.3f}".format(
          frame, sps, stats['actor_loss'], stats["mean_return"]))
        
        if frame > 0 and frame % (args.eval_every*T*B) == 0:
            utils.validate(model, render=args.render)
            model.train()


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--mode", default="train", choices=["train", "valid",], 
                    help="training or validation mode")
parser.add_argument("--total_frames", default=1000000, type=int, 
                    help="total environment frames to train for")
parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, 
                    help="unroll length (time dimension)")
parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")
parser.add_argument("--gamma", default=0.90, type=float, help="discount factor")
parser.add_argument("--entropy_coef", default=0.9, type=float, help="entropy coefficient")

if __name__ == "__main__":
    torch.manual_seed(1729)
    np.random.seed(1729)
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
        model = Actor(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args)
