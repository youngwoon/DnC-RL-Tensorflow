#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
import argparse
import sys
sys.path.insert(0, '../../gym')
sys.path.insert(0, '../../')
import os.path as osp
import os
import time
import logging

from mpi4py import MPI
import gym

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
from baselines.common.atari_wrappers import FrameStack_Mujoco


def train(env_id,
          rank,
          environment_args,
          stacked_obs,
          num_hidden_units,
          max_iters,
          checkpoint_dir,
          log_dir,
          timesteps_per_batch,
          render,
          seed):

    sess = U.single_threaded_session()
    sess.__enter__()

    if rank == 0:
        logger.configure()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    if environment_args is not None:
        try:
            env.unwrapped.set_environment_config(environment_args)
        except:
            print("Can't set the configuration to the environment!")

    if rank == 0:
        with open(osp.join(checkpoint_dir, "args.txt"), "a") as f:
            f.write("\nEnvironment argument:\n")
            for k, v in env.unwrapped._config.items():
                f.write("{}: {}\n".format(k, v))

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                         hid_size=num_hidden_units, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)),
                        allow_early_resets=True)

    # Support the stacked the frames
    env = FrameStack_Mujoco(env, stacked_obs)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env,
                   policy_fn,
                   checkpoint_dir,
                   log_dir,
                   render=render,
                   timesteps_per_batch=timesteps_per_batch,
                   max_kl=0.01,
                   cg_iters=10,
                   cg_damping=0.1,
                   max_iters=max_iters,
                   gamma=0.99,
                   lam=0.98,
                   vf_iters=5,
                   vf_stepsize=1e-3)
    env.close()


def evaluate(env_id,
             environment_args,
             stacked_obs,
             num_hidden_units,
             load_model_path,
             timesteps_per_batch,
             video_prefix,
             render,
             record,
             seed,
             info_list):

    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)

    if environment_args is not None:
        try:
            env.unwrapped.set_environment_config(environment_args)
        except:
            print("Can't set the configuration to the environment!")

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                         hid_size=num_hidden_units, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)),
                        allow_early_resets=True)

    # Support the stacked the frames
    env = FrameStack_Mujoco(env, stacked_obs)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.evaluate(env,
                      policy_fn,
                      timesteps_per_batch,
                      load_model_path,
                      video_prefix,
                      record=record,
                      render=render,
                      info_list=info_list,
                      gamma=0.99,
                      lam=0.98,)
    env.close()


def encode_args(args_str):
    args_dict = {}
    args_list = args_str.split("/")
    for args in args_list:
        k, v = args.split('-')
        args_dict.update({k: float(v)})
    return args_dict


def main():
    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # env
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--environment_args', type=str, default=None)
    parser.add_argument('--stacked_obs', type=int, default=1)

    # training
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--timesteps_per_batch', type=int, default=10000)
    parser.add_argument('--num_hidden_units', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=int(500))
    parser.add_argument('--is_train', type=str2bool, default=True)

    # saving
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--video_prefix', type=str, default=None)

    # other
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--record', type=str2bool, default=False)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()

    env_name = args.env.split('-')[0]
    if args.environment_args is not None:
        if args.prefix is None:
            args.prefix = args.environment_args.replace('/', '&')
        else:
            args.prefix += "." + args.environment_args.replace('/', '&')
        env_name = env_name + '.{}'.format(args.prefix)
        args.environment_args = encode_args(args.environment_args)
        if rank == 0:
            print("Using args: {}".format(args.__dict__))
    else:
        if args.prefix is not None:
            env_name = env_name + '.{}'.format(args.prefix)

    args.checkpoint_dir = osp.join(args.log_dir, env_name)
    args.log_dir = osp.join(args.log_dir, env_name)

    if args.is_train:
        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(osp.join(args.checkpoint_dir, "args.txt"), "w") as f:
                f.write("Date and Time:\n")
                f.write(time.strftime("%d/%m/%Y\n"))
                f.write(time.strftime("%H:%M:%S\n\n"))
                for k, v in args.__dict__.items():
                    if k != "environment_args":
                        f.write("{}: {}\n".format(k, v))
        train(args.env, rank,
              timesteps_per_batch=args.timesteps_per_batch,
              environment_args=args.environment_args,
              stacked_obs=args.stacked_obs,
              num_hidden_units=args.num_hidden_units,
              max_iters=args.max_iters,
              checkpoint_dir=args.checkpoint_dir,
              log_dir=args.log_dir,
              render=args.render,
              seed=args.seed)
    else:
        # information to analyze
        info_list = [""]
        evaluate(args.env,
                 environment_args=args.environment_args,
                 stacked_obs=args.stacked_obs,
                 num_hidden_units=args.num_hidden_units,
                 load_model_path=args.load_model_path,
                 timesteps_per_batch=args.timesteps_per_batch,
                 video_prefix=args.video_prefix,
                 record=args.record,
                 render=args.render,
                 seed=args.seed,
                 info_list=info_list)

if __name__ == '__main__':
    main()
