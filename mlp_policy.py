import tensorflow as tf
import numpy as np
import gym

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U

import ops


class MlpPolicy(object):
    def __init__(self, name, env, config):
        # args
        self.name = name

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

        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name
            self._build()
            self.var_list = [v for v in self.get_variables() if 'vf' not in v.name]

    def _build(self):
        ac_space = self._ac_space
        num_hid_layers = self._num_hid_layers
        hid_size = self._hid_size
        gaussian_fixed_var = self._gaussian_fixed_var

        # obs
        self._obs = {}
        for ob_name, ob_shape in self._ob_shape.items():
            self._obs[ob_name] = U.get_placeholder(
                name="ob_{}".format(ob_name),
                dtype=tf.float32,
                shape=[None] + self._ob_shape[ob_name])

        # obs normalization
        self.ob_rms = {}
        for ob_name in self.ob_type:
            with tf.variable_scope("ob_rms_{}".format(ob_name)):
                self.ob_rms[ob_name] = RunningMeanStd(shape=self._ob_shape[ob_name])
        obz = [(self._obs[ob_name] - self.ob_rms[ob_name].mean) / self.ob_rms[ob_name].std
               for ob_name in self.ob_type]
        obz = [tf.clip_by_value(ob, -5.0, 5.0) for ob in obz]
        obz = tf.concat(obz, -1)

        # value function
        with tf.variable_scope('vf'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.layers.dense(last_out, hid_size, name="fc%i" % (i+1),
                                    kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name="final",
                                         kernel_initializer=U.normc_initializer(1.0))[:,0]

        # policy
        self.pdtype = pdtype = make_pdtype(ac_space)
        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.layers.dense(last_out, hid_size, name="fc%i" % (i+1),
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
        self.pd = pdtype.pdfromflat(pdparam)

        # sample action
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.obs = [self._obs[ob_name] for ob_name in self.ob_type]
        self._act = U.function([stochastic] + self.obs, [ac, self.vpred])
        self._value = U.function([stochastic] + self.obs, self.vpred)

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
