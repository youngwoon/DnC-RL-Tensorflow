#!/usr/bin/env python
import sys
sys.path.insert(0, 'gym')
import os
import logging
import signal
import threading

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


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args):
    sess = U.single_threaded_session()
    sess.__enter__()

    # setting envs and networks
    with tf.device("/cpu:0"):
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
            env.unwrapped.set_context(i)
            network = MlpPolicy(i, 'local_%d' % i, env, args)
            old_network = MlpPolicy(i, 'old_local_%d' % i, env, args)

            envs.append(env)
            networks.append(network)
            old_networks.append(old_network)

        for i in range(args.num_workers):
            runner = Runner(env, network, config=args)
            trainer = LocalTrainer(i, envs[i], runner,
                                   networks[i], old_networks[i], networks,
                                   global_network, args)
            trainers.append(trainer)

    # summaries
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(tf.global_variables())

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.get_variable_scope().name)
    logger.info('All vars:')
    for v in var_list:
        logger.info('  {} ({}, {})'.format(
                v.name, v.dtype.name, 'x'.join([str(size) for size in v.get_shape()])
        ))

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  {} ({}, {})'.format(
                v.name, v.dtype.name, 'x'.join([str(size) for size in v.get_shape()])
        ))

    def init_fn(sess):
        logger.info("Initializing all parameters.")
        sess.run(init_all_op)

    log_dir = os.path.join(args.log_dir, args.env)
    summary_writer = tf.summary.FileWriter(log_dir)
    logger.info("Events directory: %s", log_dir)

    summary_name = global_trainer.summary_name.copy()
    for trainer in trainers:
        summary_name.extend(trainer.summary_name)
    ep_stats = stats(summary_name)

    # start training
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
        coord = tf.train.Coordinator()
        worker_threads = []
        for trainer in trainers:
            worker = lambda: trainer.update(sess)
            t = threading.Thread(target=(worker))
            worker_threads.append(t)

    global_step = sess.run(trainer.global_step)
    logger.info("Starting training at step=%d", global_step)

    pbar = tqdm.trange(global_step, args.T, total=args.T, initial=global_step)
    for step in pbar:
        for trainer in trainers:
            trainer.init_network()

        global_rollouts = []
        for _ in range(args.R):
            if args.threading:
                for t in worker_threads:
                    t.start()
                coord.join(worker_threads)
            else:
                for trainer in trainers:
                    trainer.generate_rollout()

            rollouts = []
            for trainer in trainers:
                rollouts.append(trainer.rollout)
            global_rollouts.extend(rollouts)
            info = {}
            for trainer in trainers:
                _info = trainer.update(sess, rollouts)
                info.update(_info)

            global_step = sess.run(trainer.global_step)
            ep_stats.add_all_summary_dict(summary_writer, info, global_step)

        if args.training_video_record:
            for trainer in trainers:
                trainer.evaluate(global_step, record=True)

        ob = np.concatenate([rollout['ob'] for rollout in global_rollouts])
        ac = np.concatenate([rollout['ac'] for rollout in global_rollouts])

        info = global_trainer.update(step, ob, ac)
        ep_stats.add_all_summary_dict(summary_writer, info, global_step)

        pbar.set_description('')

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
