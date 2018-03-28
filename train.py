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
        global_network = MlpPolicy('global', global_env, args)
        global_runner = Runner(global_env, global_network, config=args)
        global_trainer = GlobalTrainer('global', global_env, global_runner, global_network, args)

        envs = []
        networks = []
        old_networks = []
        trainers = []
        for i in range(args.num_workers):
            env = gym.make(args.env)
            env.unwrapped.set_context(i)
            network = MlpPolicy('local_%d' % i, env, args)
            old_network = MlpPolicy('old_local_%d' % i, env, args)
            runner = Runner(env, network, config=args)
            trainer = LocalTrainer('local_%d' % i, env, runner,
                                   network, old_network, global_network, args)

            envs.append(env)
            networks.append(network)
            old_networks.append(old_network)
            trainers.append(trainer)

    # summaries
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(tf.global_variables())

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

    # start training
    coord = tf.train.Coordinator()
    if args.load_model_path:
        logger.info('Load models from checkpoint...')
        pass # TODO: load checkpoint from args.load_model_path
    else:
        sess.run(tf.global_variables_initializer())

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

        rollouts = []
        for _ in range(args.R):
            if args.threading:
                for t in worker_threads:
                    t.start()
                coord.join(worker_threads)
            else:
                for trainer in trainers:
                    trainer.update(sess)

            for trainer in trainers:
                rollouts.append(trainer.rollout)

        ob = np.concatenate([rollout['ob'] for rollout in rollouts])
        ac = np.concatenate([rollout['ac'] for rollout in rollouts])

        global_trainer.update(step, ob, ac)

        pbar.set_description('')

    for env in envs:
        env.close()


def main(_):
    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    args = config.argparser()
    run(args)


if __name__ == "__main__":
    tf.app.run()
