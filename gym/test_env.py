import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')

import argparse
import gym
import imageio
import numpy as np
import time
import ipdb

from baselines.common.atari_wrappers import FrameStack_Mujoco


def argsparser():
    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser("Testing code")
    parser.add_argument('--env', type=str, default='AntBandits-v1')
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--stacked_obs', type=str2bool, default=False)
    parser.add_argument('--save_img', type=str2bool, default=False)
    parser.add_argument('--sleep', type=int, default=0)
    args = parser.parse_args()
    return args


def random_play_one_episode(env, render=False, save_img=False, sleep=0, random_len=0):

    ob = env.reset()
    total_rew = 0
    total_len = 0
    while True:
        if save_img:
            imageio.imsave('{}.png'.format(total_len), env.unwrapped.get_visual_observation())
        if total_len < random_len:
            ob, rew, done, _ = env.step(env.action_space.sample())
        else:
            ob, rew, done, _ = env.step(np.zeros(env.action_space.sample().shape))

        sub_obs = env.unwrapped.convert_ob_to_sub_ob(ob)
        if render:
            env.render()
            time.sleep(sleep)
        total_rew += rew
        total_len += 1
        #if done:
        #    if total_len < random_len:
        #        print("It's random action\'s fault")
        #    break
    env.close()
    print('Total reward: {}, Total length: {}'.format(total_rew, total_len))


def main(args):
    env = gym.make(args.env)
    if args.stacked_obs:
        env = FrameStack_Mujoco(env, 4)
    ob = env.reset()
    sub_obs = env.unwrapped.convert_ob_to_sub_ob(ob)
    print("observation space: {}, action space: {}".format(env.observation_space, env.action_space))
    print("sub policies observation space: {}, action space: {}".format(env.unwrapped.unwrapped.sub_policy_observation_space, env.action_space))
    print("[For verifying] sub_policy observation shape: {}".format(sub_obs.shape))
    while True:
        random_play_one_episode(env, args.render, args.save_img, args.sleep)


if __name__ == '__main__':
    args = argsparser()
    main(args)
