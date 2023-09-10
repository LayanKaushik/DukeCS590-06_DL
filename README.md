# DukeCS590-06_DL

## Advantage Actor-Critic (A2C) to play the Ms. Pacman Atari game

# Improvements and Modifications

## Learning Algorithm Enhancement

The Advantage Actor-Critic (A2C) algorithm has been implemented, involving the creation of two primary models: Actor and Actor-Critic. The architecture of both these models has been further refined in Model Refinement. The Actor model uses a neural network to output a probability distribution over possible actions for a given state, while the Actor-Critic model outputs a single scalar value, indicating the expected reward for that state.

To enhance the actor's performance, the policy gradient step was adjusted to compute the advantage function utilizing the actor-critic neural network. The advantage function determines the superiority of one action over the average action for a specific state. Leveraging the advantage function for decision-making allows for quicker convergence and superior results.

The advantage function for the actor is expressed as:

advantage = rewards + gamma x Actor_Critic(next-states) - Actor_Critic(current-state)

Here:
- `rewards`: Immediate reward from current action-state pair.
- `gamma`: Discount factor for future rewards.
- `Actor_Critic(next-states)`: Expected value of next state via the actor-critic model.
- `Actor_Critic(current-state)`: Expected value of current state via the actor-critic model.

The losses for both actor and critic are computed as:

actor_loss = -log(probability_of_action) x advantage
critic_loss = (advantage - Actor_Critic(current-state))^2
total_loss = actor_loss + critic_loss


To further optimize, the total loss is backpropagated. Additionally, a penalty of -10 is introduced for death events, motivating the agent to maximize survival time.

## Model Refinement

Performance enhancement of the A2C algorithm was initiated by preprocessing the observations to filter and simplify input data. Observations were cropped, downsampled, and converted to grayscale. Following this, modifications were made to the actor and actor-critic CNN models.

Several features were incorporated, including batch normalization and pooling layers. The activation function was switched to ReLU due to its computational efficiency and extensive testing in various applications. An entropy coefficient was introduced to facilitate exploration, aiming to help the agent avoid local minima. However, this change did not significantly influence the model's performance.

## Execution Instructions

To run the provided code:

1. Download and store the three python files in a directory.
2. Navigate to that directory in the terminal.
3. Execute the following command:

python -u hw4_agent_final.py --mode valid --load_path model_final.pt --render

# Baseline validation

Steps taken over each of 1 episodes: 169

Total return after 1 episodes: 210.000

![Figure_1_baseline](https://user-images.githubusercontent.com/24253218/235557485-c8d21a10-7d5a-478a-9f05-43e3ffa17be2.png)

# References:

https://arxiv.org/abs/1312.5602

@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}

@online{huggingface2023deep-rl,
  author    = {Hugging Face},
  title     = {Deep RL Course},
  year      = {2023},
  url       = {https://huggingface.co/learn/deep-rl-course/unit0/introduction?fw=pt}
}

@online{tabor2023actor-critic,
  author    = {Phil Tabor},
  title     = {Actor-Critic Methods: Paper to Code},
  year      = {2023},
  url       = {https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master}
}

@online{nguyen2023cleanrl,
  author    = {Thanh Nguyen},
  title     = {CleanRL: Clean and Simple Reinforcement Learning},
  year      = {2023},
  url       = {https://github.com/vwxyzjn/cleanrl}
}


