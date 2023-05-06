import gymnasium as gym
import torch
from test_agent import DQNAgent, train
from test_utils import preprocess_observation, validate_agent

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--mode", default="train", choices=["train", "valid"], 
                    help="training or validation mode")
parser.add_argument("--total_frames", default=1000000, type=int, 
                    help="total environment frames to train for")
parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, 
                    help="unroll length (time dimension)")
parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")


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
        model = DQNAgent(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args)

