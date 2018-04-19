import argparse
import os


def argparser():
    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser("Divide-and-Conquer RL",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--num_contexts', default=4, type=int,
                        help="Number of contexts (do not change!)")
    parser.add_argument('-e', '--env', type=str, default="JacoPick-v1",
                        help="Environment id")
    parser.add_argument('-l', '--log_dir', type=str, default="log",
                        help="Log directory path")

    # model parameters
    parser.add_argument('--method', type=str, default='dnc', choices=['trpo', 'dnc'],
                        help="Training method")

    # training
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help="Training or evaluation")
    parser.add_argument('--T', type=int, default=100,
                        help="Number of outer training loop")
    parser.add_argument('--R', type=int, default=100,
                        help="Distillation period")
    parser.add_argument('--num_rollouts', type=int, default=20000,
                        help="Length of rollouts for a single update")
    parser.add_argument('--load_model_path', type=str, default=None,
                        help="Path of the pre-trained model to load")

    # local network (trpo)
    parser.add_argument('--num_hid_layers', type=int, default=2,
                        help="Number of MLP layers in policy network")
    parser.add_argument('--hid_size', type=int, nargs='*', default=[64, 64],
                        help="Hidden layer size")
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['relu', 'elu', 'tanh'],
                        help="Non-linear activation function")
    parser.add_argument('--fixed_var', type=str2bool, default=True,
                        help="Variance of stochastic policy is fixed over observation")
    parser.add_argument('--obs_norm', type=str, default='predefined',
                        choices=['no', 'learn', 'predefined'],
                        help="Normalize observation")

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_damping', type=float, default=0.1)
    parser.add_argument('--vf_stepsize', type=float, default=1e-3)
    parser.add_argument('--vf_iters', type=int, default=5)
    parser.add_argument('--vf_batch_size', type=int, default=64)
    parser.add_argument('--ent_coeff', type=float, default=0)
    parser.add_argument('--divergence_coeff', type=float, default=1e-4)

    # global network
    parser.add_argument('--global_stepsize', type=float, default=1e-2)
    parser.add_argument('--global_iters', type=int, default=20)
    parser.add_argument('--global_max_grad_norm', type=float, default=10.0)
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--global_vf', type=str2bool, default=False,
                        help="Distill value network")

    # log, record, ckpt
    parser.add_argument('--ckpt_save_step', type=int, default=1,
                        help="Checkpoint will be saved every 'CKPT_SAVE_STEP' global steps")
    parser.add_argument('--training_video_record', type=str2bool, default=True,
                        help="Save evaluation videos every global step")

    # misc
    parser.add_argument('--prefix', type=str, default=None,
                        help="Prefix of the log dir")
    parser.add_argument('--render', type=str2bool, default=False,
                        help="Render screen")
    parser.add_argument('--record', type=str2bool, default=False,
                        help="Record screen")
    parser.add_argument('--video_prefix', type=str, default=None,
                        help="Prefix of video files")
    parser.add_argument('--video_format', type=str, default='mp4',
                        choices=['mp4', 'gif'],
                        help="Format of video files")
    parser.add_argument('--seed', type=int, default=0,
                        help="RNG seed")
    parser.add_argument('--debug', type=str2bool, default=False,
                        help="Print out debugging info")
    parser.add_argument('--display', type=str, default=None,
                        help="Set 'DISPLAY' environment variable for virtual screen")

    args = parser.parse_args()
    if args.training_video_record and args.display is not None:
        os.environ["DISPLAY"] = ":" + args.display
    return args
