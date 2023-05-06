import logging
import torch
import numpy as np
import gymnasium as gym
from torchvision.transforms import Resize, ToPILImage, ToTensor

logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

try:
    import matplotlib.pyplot as plt
    can_render = True
except:
    logging.warning("Cannot import matplotlib; will not attempt to render")
    can_render = False


def preprocess_observation(obs):
    """
    obs - a 210 x 160 x 3 ndarray representing an atari frame
    returns:
      a 3 x 210 x 160 normalized pytorch tensor
    """
    return torch.from_numpy(obs).permute(2, 0, 1)/255.0

def validate(model, render=False, nepisodes=1):
    assert hasattr(model, "choose_greedy_action")
    torch.manual_seed(590060)
    np.random.seed(590060)
    model.eval()

    render = render and can_render

    if render:
        nepisodes = 1
        fig, ax = plt.subplots(1, 1)

    total_reward = 0
    steps_alive = []
    for i in range(nepisodes):
        env = gym.make("ALE/MsPacman-v5")
        obs = env.reset(seed=590060+i)[0]
        if render:
            im = ax.imshow(obs)
        observation = preprocess_observation( # 1 x 1 x ic x iH x iW
            obs).unsqueeze(0).unsqueeze(0)
        prev_state = None
        step = 0
        # play until the agent dies or we exceed 50000 observations
        while env.ale.lives() == 3 and step < 50000:
            action, prev_state = model.choose_greedy_action(observation, prev_state)
            env_output = env.step(action)
            total_reward += env_output[1]
            if render:
                img = env_output[0]
                im.set_data(img)
                fig.canvas.draw_idle()
                plt.pause(0.1)
            observation = preprocess_observation(
                env_output[0]).unsqueeze(0).unsqueeze(0)
            step += 1
        steps_alive.append(step)

    logging.info("Steps taken over each of {:d} episodes: {}".format(
        nepisodes, ", ".join(str(step) for step in steps_alive)))
    logging.info("Total return after {:d} episodes: {:.3f}".format(nepisodes, total_reward))




# class RepeatActionAndMaxFrame(gym.Wrapper):
#     """ modified from:
#         https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
#     """
#     def __init__(self, env=None, repeat=4):
#         super(RepeatActionAndMaxFrame, self).__init__(env)
#         self.repeat = repeat
#         self.shape = env.observation_space.low.shape
#         self.frame_buffer = np.zeros_like((2,self.shape))

#     def step(self, action):
#         t_reward = 0.0
#         done = False
#         for i in range(self.repeat):
#             obs, reward, done, info = self.env.step(action)
#             t_reward += reward
#             idx = i % 2
#             self.frame_buffer[idx] = obs
#             if done:
#                 break

#         max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
#         return max_frame, t_reward, done, info

#     def reset(self):
#         obs = self.env.reset()
#         self.frame_buffer = np.zeros_like((2,self.shape))
#         self.frame_buffer[0] = obs
#         return obs

# class PreprocessFrame(gym.ObservationWrapper):
#     def __init__(self, shape, env=None):
#         super(PreprocessFrame, self).__init__(env)
#         self.shape=(shape[2], shape[0], shape[1])
#         self.observation_space = gym.spaces.Box(low=0, high=1.0,
#                                               shape=self.shape,dtype=np.float32)
#     def observation(self, obs):
#         new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         resized_screen = cv2.resize(new_frame, self.shape[1:],
#                                     interpolation=cv2.INTER_AREA)

#         new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
#         new_obs = np.swapaxes(new_obs, 2,0)
#         new_obs = new_obs / 255.0
#         return new_obs

# class StackFrames(gym.ObservationWrapper):
#     def __init__(self, env, n_steps):
#         super(StackFrames, self).__init__(env)
#         self.observation_space = gym.spaces.Box(
#                              env.observation_space.low.repeat(n_steps, axis=0),
#                              env.observation_space.high.repeat(n_steps, axis=0),
#                              dtype=np.float32)
#         self.stack = collections.deque(maxlen=n_steps)

#     def reset(self):
#         self.stack.clear()
#         observation = self.env.reset()
#         for _ in range(self.stack.maxlen):
#             self.stack.append(observation)

#         return np.array(self.stack).reshape(self.observation_space.low.shape)

#     def observation(self, observation):
#         self.stack.append(observation)
#         obs = np.array(self.stack).reshape(self.observation_space.low.shape)

#         return obs

# def make_env(env_name, shape=(84,84,1), skip=4):
#     env = gym.make(env_name)
#     env = RepeatActionAndMaxFrame(env, skip)
#     env = PreprocessFrame(shape, env)
#     env = StackFrames(env, skip)

#     return env