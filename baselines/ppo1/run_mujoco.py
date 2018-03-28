#!/usr/bin/env python
import argparse
import sys
sys.path.insert(0, '../gym')
sys.path.insert(0, '../../')
import os.path as osp
import gym
import logging
import ipdb

from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines import logger

def train(env_id, environment_args, num_timesteps, checkpoint_dir, log_dir, seed):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    if environment_args is not None:
        try:
            env.unwrapped.set_environment_config(environment_args)
        except:
            print("Can't set the configuration to the environment!")

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, checkpoint_dir, log_dir,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def evaluate(env_id, num_timesteps, load_model_path, video_prefix, record, render, seed):

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    pposgd_simple.evaluate(env, policy_fn, load_model_path, video_prefix,
                           timesteps_per_batch=1024, record=record, render=render)
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
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--environment_args', type=str, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--record', type=str2bool, default=False)
    parser.add_argument('--video_prefix', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--log_dir', type=str, default='log')
    args = parser.parse_args()
    logger.configure()

    env_name = args.env.split('-')[0]

    if args.environment_args is not None:
        args.environment_args = encode_args(args.environment_args)
        if args.prefix is None:
            args.prefix = ""
        for k, v in args.environment_args.items():
            args.prefix += ".{}-{}".format(k, v)
        env_name = env_name + '{}'.format(args.prefix)
    else:
        if args.prefix is not None:
            env_name = env_name + '.{}'.format(args.prefix)

    args.checkpoint_dir = osp.join(args.checkpoint_dir, env_name)
    args.log_dir = osp.join(args.log_dir, env_name)

    if args.is_train:
        train(args.env,
              environment_args=args.environment_args,
              num_timesteps=args.num_timesteps,
              checkpoint_dir=args.checkpoint_dir,
              log_dir=args.log_dir,
              seed=args.seed)
    else:
        evaluate(args.env, num_timesteps=args.num_timesteps, load_model_path=args.load_model_path,
                 video_prefix=args.video_prefix, record=args.record, render=args.render, seed=args.seed)


if __name__ == '__main__':
    main()
