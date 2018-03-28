import argparse
import os


def argparser():
    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser("Divide-and-Conquer RL",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-w', '--num-workers', default=4, type=int,
                        help="Number of workers")
    parser.add_argument('-e', '--env', type=str, default="JacoPick-v1",
                        help="Environment id")
    parser.add_argument('--env_args', type=str, default=None,
                        help="Environment arugments")
    parser.add_argument('-l', '--log-dir', type=str, default="log",
                        help="Log directory path")
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help="Print out commands rather than executing them")

    # model parameters
    parser.add_argument('--method', type=str, default='dnc',
                        choices=['trpo', 'dnc'], help="training method")

    # vanilla rl
    parser.add_argument('--num_hid_layers', type=int, default=2)
    parser.add_argument('--hid_size', type=int, default=32)
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--fixed_var', type=str2bool, default=True)

    # training
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--threading', type=str2bool, default=False)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--T', type=int, default=1000,
                        help="Number of training loop")
    parser.add_argument('--R', type=int, default=100,
                        help="Distillation period")
    parser.add_argument('--num_rollouts', type=int, default=1000)
    parser.add_argument('--num_batches', type=int, default=32)
    parser.add_argument('--num_trans_batches', type=int, default=256)

    # local network (trpo)
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_damping', type=float, default=0.1)
    parser.add_argument('--vf_stepsize', type=float, default=1e-3)
    parser.add_argument('--vf_iters', type=int, default=5)
    parser.add_argument('--entcoeff', type=float, default=1e-2)
    parser.add_argument('--optim_epochs', type=int, default=10)
    parser.add_argument('--optim_stepsize', type=float, default=3e-4)
    parser.add_argument('--optim_batchsize', type=int, default=64)

    # global network
    parser.add_argument('--global_stepsize', type=float, default=1e-3)
    parser.add_argument('--global_max_grad_norm', type=float, default=10.0)

    # log, record, ckpt
    parser.add_argument('--ckpt_save_step', type=int, default=100)
    parser.add_argument('--training_video_record', type=str2bool, default=True)
    parser.add_argument('--record_script_cmd_path', type=str, default=None)

    # misc
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--record', type=str2bool, default=False)
    parser.add_argument('--video_prefix', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=0,
                        help="RNG seed")
    parser.add_argument('--debug', type=str2bool, default=False)

    args = parser.parse_args()
    if args.training_video_record:
        os.environ["DISPLAY"] = ":0"
    return args
