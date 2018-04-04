#!/usr/bin/env python
import sys
sys.path.insert(0, 'gym')
import os
import logging
import signal
import threading
import pipes
import time
from six.moves import shlex_quote

import tqdm
import tensorflow as tf
import numpy as np
import gym

import baselines.common.tf_util as U
from baselines.statistics import stats

import config
from mlp_policy import MlpPolicy
from global_trainer import GlobalTrainer
from local_trainer import LocalTrainer
from runner import Runner


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_info(args):
    # save command
    train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
    logger.info('\n{}\nTraining command:\n{}\n{}\n'.format('*'*80, train_cmd, '*'*80))
    with open(os.path.join(args.log_dir, "cmd.txt"), "a+") as f:
        f.write(train_cmd)

    # save argument list
    logger.info('Save argument list to {}/args.txt'.format(args.log_dir))
    args_lines = ["Date and Time:", time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S\n")]
    args_lines += ["{}: {}".format(k, v) for k, v in args.__dict__.items()]
    args_lines = '\n'.join(args_lines)
    with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
        f.write(args_lines)

    # save code revision
    logger.info('Save git commit and diff to {}/git.txt'.format(args.log_dir))
    cmds = ["echo `git rev-parse HEAD` >> {}".format(
                shlex_quote(os.path.join(args.log_dir, 'git.txt'))),
            "git diff >> {}".format(
                shlex_quote(os.path.join(args.log_dir, 'git.txt')))]
    os.system("\n".join(cmds))


def print_variables():
    def print_format(var_list):
        for v in var_list:
            logger.info('  {} ({}, {})'.format(
                    v.name, v.dtype.name, 'x'.join([str(size) for size in v.get_shape()])
            ))

    logger.info('All vars:')
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.get_variable_scope().name)
    print_format(var_list)

    logger.info('Trainable vars:')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)
    print_format(var_list)


def run(args):
    sess = U.single_threaded_session()
    sess.__enter__()

    if args.method == 'trpo':
        args.num_workers = 1

    # setting envs and networks
    global_env = gym.make(args.env)
    global_network = MlpPolicy(0, 'global', global_env, args)
    global_runner = Runner(global_env, global_network, config=args)
    global_trainer = GlobalTrainer('global', global_env, global_runner, global_network, args)

    envs = []
    networks = []
    old_networks = []
    trainers = []
    for i in range(args.num_workers):
        env = gym.make(args.env)
        if args.method == 'dnc':
            env.unwrapped.set_context(i)
        network = MlpPolicy(i, 'local_%d' % i, env, args)
        old_network = MlpPolicy(i, 'old_local_%d' % i, env, args)

        envs.append(env)
        networks.append(network)
        old_networks.append(old_network)

    for i in range(args.num_workers):
        runner = Runner(env, networks[i], config=args)
        trainer = LocalTrainer(i, envs[i], runner,
                               networks[i], old_networks[i], networks,
                               global_network, args)
        trainers.append(trainer)

    # summaries
    print_variables()

    exp_name = '{}_{}'.format(args.env, args.method)
    if args.prefix:
        exp_name = '{}_{}'.format(exp_name, args.prefix)
    args.log_dir = os.path.join(args.log_dir, exp_name)
    logger.info("Events directory: %s", args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    write_info(args)

    if args.is_train:
        summary_writer = tf.summary.FileWriter(args.log_dir)
        summary_name = global_trainer.summary_name.copy()
        for trainer in trainers:
            summary_name.extend(trainer.summary_name)
        ep_stats = stats(summary_name)

    # initialize model
    if args.load_model_path:
        logger.info('Load models from checkpoint...')
        def load_model(load_model_path, var_list=None):
            if os.path.isdir(load_model_path):
                ckpt_path = tf.train.latest_checkpoint(load_model_path)
            else:
                ckpt_path = load_model_path
            U.load_state(ckpt_path, var_list)
            return ckpt_path
        load_model(args.load_model_path)
    else:
        sess.run(tf.global_variables_initializer())

    if args.threading:
        raise NotImplementedError('Multi-threading is not implemented.')
        coord = tf.train.Coordinator()

    global_step = sess.run(trainers[0].global_step)
    logger.info("Starting training at step=%d", global_step)

    pbar = tqdm.trange(global_step, args.T, total=args.T, initial=global_step)
    for step in pbar:
        for trainer in trainers:
            trainer.init_network()

        for _ in range(args.R):
            # get rollouts
            if args.threading:
                rollout_threads = []
                for trainer in trainers:
                    worker = lambda: trainer.generate_rollout(sess)
                    t = threading.Thread(target=(worker))
                    t.start()
                    rollout_threads.append(t)
                coord.join(rollout_threads)
            else:
                for trainer in trainers:
                    trainer.generate_rollout(sess)

            rollouts = []
            for trainer in trainers:
                rollouts.append(trainer.rollout)

            # update local policies
            info = {}
            for trainer in trainers:
                _info = trainer.update(sess, rollouts)
                info.update(_info)

            global_step = sess.run(trainer.global_step)
            ep_stats.add_all_summary_dict(summary_writer, info, global_step)

        # update global policy using the last rollouts
        global_info = info
        if args.method == 'dnc':
            ob = np.concatenate([rollout['ob'] for rollout in rollouts])
            ac = np.concatenate([rollout['ac'] for rollout in rollouts])
            info = global_trainer.update(step, ob, ac)
            global_info.update(info)
        else:
            trainer.copy_network()

        # evaluate local policies
        for trainer in trainers:
            trainer.evaluate(global_step, record=args.training_video_record)

        # evaluate global policy
        info = global_trainer.summary(step)
        global_info.update(info)
        ep_stats.add_all_summary_dict(summary_writer, global_info, global_step)
        pbar.set_description(
            '[step {}] reward {} length {} success {}'.format(
                step, global_info['reward'], global_info['length'], global_info['success'])
        )

    for env in envs:
        env.close()


def main(_):
    def shutdown(signal, frame):
        logger.warning('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    args = config.argparser()
    run(args)


if __name__ == "__main__":
    tf.app.run()
