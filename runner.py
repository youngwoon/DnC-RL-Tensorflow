from collections import defaultdict

import numpy as np
import cv2


def render_frame(env, length, ret, sub_name, render, record=False):
    if not render and not record:
        return None
    raw_img = env.unwrapped.get_visual_observation()
    raw_img = np.asarray(raw_img, dtype=np.uint8).copy()
    cv2.putText(raw_img, '{:4d} {:.2f} {}'.format(length, ret, sub_name),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 128, 128), 2, cv2.LINE_AA)
    if render:
        raw_img = cv2.resize(raw_img, (1000, 1000))
        cv2.imshow(env.spec.id, raw_img)
        cv2.waitKey(1)
    return raw_img if record else None


class Rollout(object):
    def __init__(self):
        self._history = defaultdict(list)

    def clear(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        return self._history


class Runner(object):
    def __init__(self, env, policy, config):
        self._config = config
        self._env = env
        self._pi = policy

        if config.render:
            cv2.namedWindow(env.spec.id)
            cv2.moveWindow(env.spec.id, 0, 0)

    def rollout(self, stochastic, training_inference=False):
        config = self._config
        env = self._env
        pi = self._pi

        record = config.record or (config.training_video_record and training_inference)
        single_episode = not config.is_train or training_inference

        if config.method == 'dnc' and not pi.name.startswith('global'):
            env.unwrapped.set_context(pi._id)
        else:
            env.unwrapped.set_context(-1)

        ob = env.reset()
        ac = env.action_space.sample()
        term = False
        rew = 0.
        cur_ep_ret = 0.
        t = 0
        cur_ep_len = 0

        # initialize history arrays
        rollout = Rollout()
        reward_info = defaultdict(list)
        ep_reward = defaultdict(list)

        # rollout
        while t < config.num_rollouts:
            ac, vpred = pi.act(stochastic, ob)
            vob = render_frame(env, cur_ep_len, cur_ep_ret, pi.name, config.render, record)
            rollout.add({'ob': ob, 'ac': ac, 'vpred': vpred, 'visual_ob': vob})

            ob, rew, term, info = env.step(ac)

            rollout.add({'rew': rew, 'term': term})
            cur_ep_ret += rew
            cur_ep_len += 1
            t += 1
            for key, value in info.items():
                reward_info[key].append(value)

            if term:
                vob = render_frame(env, cur_ep_len, cur_ep_ret, pi.name, config.render, record)
                rollout.add({'visual_ob': vob, 'ep_reward': cur_ep_ret, 'ep_length': cur_ep_len})
                for key, value in reward_info.items():
                    if isinstance(value[0], (int, float)):
                        if key.endswith('mean'):
                            ep_reward[key[:-5]].append(np.mean(value))
                        elif key.endswith('sum'):
                            ep_reward[key[:-4]].append(np.sum(value))
                        else:
                            ep_reward[key].append(np.sum(value))

                if single_episode:
                    break

                reward_info = defaultdict(list)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()

        dicti = rollout.get()
        dicti["next_vpred"] = vpred * (1 - term)
        for key, value in ep_reward.items():
            dicti.update({"ep_{}".format(key): value})
        return {key: np.copy(val) for key, val in dicti.items()}

    def add_advantage(self, seg, gamma, lam):
        term = seg["term"]
        rew = seg["rew"]
        vpred = np.append(seg["vpred"], seg["next_vpred"])
        T = len(rew)
        seg["adv"] = gaelam = np.empty(T, 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - term[t]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

        assert np.isfinite(seg["vpred"]).all()
        assert np.isfinite(seg["next_vpred"]).all()
        assert np.isfinite(seg["adv"]).all()
