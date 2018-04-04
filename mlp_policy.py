import tensorflow as tf
import numpy as np
import gym

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U

import ops


class MlpPolicy(object):
    def __init__(self, id, name, env, config):
        # args
        self._id = id
        self.name = name
        self._config = config

        # training
        self._hid_size = config.hid_size
        self._num_hid_layers = config.num_hid_layers
        self._gaussian_fixed_var = config.fixed_var
        self._activation = ops.activation(config.activation)

        # properties
        self._ob_shape = env.unwrapped.ob_shape
        self.ob_type = sorted(env.unwrapped.ob_type)

        self._env = env.unwrapped
        self._ob_space = np.sum([np.prod(ob) for ob in self._ob_shape.values()])
        self._ac_space = env.unwrapped.action_space

        # obs normalization
        if self._config.obs_norm:
            self.ob_rms = {}
            for ob_name in self.ob_type:
                with tf.variable_scope("ob_rms_{}".format(ob_name), reuse=tf.AUTO_REUSE):
                    self.ob_rms[ob_name] = RunningMeanStd(shape=self._ob_shape[ob_name])

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.scope = tf.get_variable_scope().name
            self._build()
            self.var_list = [v for v in self.get_variables() if 'vf' not in v.name]

    def _build(self):
        ac_space = self._ac_space
        num_hid_layers = self._num_hid_layers
        hid_size = self._hid_size
        gaussian_fixed_var = self._gaussian_fixed_var
        if not isinstance(hid_size, list):
            hid_size = [hid_size]
        if len(hid_size) != num_hid_layers:
            hid_size += [hid_size[-1]] * (num_hid_layers - len(hid_size))

        self.obs = []
        self.pds = []

        for j in range(self._config.num_workers):
            # obs
            _ob = {}
            for ob_name, ob_shape in self._ob_shape.items():
                _ob[ob_name] = U.get_placeholder(
                    name="ob_{}/from_{}".format(ob_name, j),
                    dtype=tf.float32,
                    shape=[None] + self._ob_shape[ob_name])

            # obs normalization
            if self._config.obs_norm:
                obz = [(_ob[ob_name] - self.ob_rms[ob_name].mean) / self.ob_rms[ob_name].std
                    for ob_name in self.ob_type]
            else:
                obz = [_ob[ob_name] for ob_name in self.ob_type]

            obz = [tf.clip_by_value(ob, -5.0, 5.0) for ob in obz]
            obz = tf.concat(obz, -1)

            # value function
            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = self._activation(
                        tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i+1),
                                        kernel_initializer=U.normc_initializer(1.0)))
                vpred = tf.layers.dense(last_out, 1, name="final",
                                        kernel_initializer=U.normc_initializer(1.0))[:,0]
                if j == self._id:
                    self.vpred = vpred

            # policy
            pdtype = make_pdtype(ac_space)
            if j == self._id:
                self.pdtype = pdtype
            with tf.variable_scope('pol', reuse=tf.AUTO_REUSE):
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = self._activation(
                        tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i+1),
                                        kernel_initializer=U.normc_initializer(1.0)))

                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name="final",
                                           kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd",
                                             shape=[1, pdtype.param_shape()[0]//2],
                                             initializer=tf.zeros_initializer())
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                else:
                    pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name="final",
                                              kernel_initializer=U.normc_initializer(0.01))

            self.obs.append([_ob[ob_name] for ob_name in self.ob_type])
            self.pds.append(pdtype.pdfromflat(pdparam))

        self.ob = self.obs[self._id]
        self.pd = self.pds[self._id]

        # sample action
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic] + self.ob, [ac, self.vpred])
        self._value = U.function([stochastic] + self.ob, self.vpred)

    def act(self, stochastic, ob):
        ob_list = self.get_ob_list(ob)
        ac, vpred = self._act(stochastic, *ob_list)
        return ac[0], vpred[0]

    def value(self, stochastic, ob):
        ob_list = self.get_ob_list(ob)
        vpred = self._value(stochastic, *ob_list)
        return vpred[0]

    def get_ob_list(self, ob):
        ob_list = []
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                ob_list.append(ob[ob_name][None])
            else:
                ob_list.append(ob[ob_name])
        return ob_list

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
