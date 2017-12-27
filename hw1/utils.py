import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import os
import argparse
import load_policy
import tensorflow.contrib.slim as slim


def smooth(a, beta=0.8):
    """curve smoothing"""
    for i in range(1, len(a)):
        a[i] = beta * a[i - 1] + (1 - beta) * a[i]

    return a


def generate_expert_data(envname, num_rollouts, max_timesteps=None, expert_policy_file=None,
                         render=False, save=True):
    """
    generate training data from expert policy
    """
    if not expert_policy_file:
        expert_policy_file = 'experts/' + envname + '.pkl'

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        results = test_run(env, num_rollouts, policy_fn, max_steps, render)
        returns = results['returns']
        observations = results['observations']
        actions = results['actions']
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        # get the space shapes
        # For Hopper-v1 ob: (11, ), action: (1, 3)
        print('shape of observations:', np.array(observations).shape)
        print('shape of actions:', np.array(actions).shape)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        # save expert_data
        if save:
            save_path = './output'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path + '/' + envname + '_expert_' +
                      str(num_rollouts) + '.pkl', 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

        return expert_data


def test_run(env, num_rollouts, policy, max_steps, render=False):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render or i + 1 == num_rollouts:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    return {'returns': returns, 'observations': observations, 'actions': actions}


class Policy():
    def __init__(self, env, layers, lr=1e-3):
        self._act_dim = env.action_space.shape[0]
        self._obs = tf.placeholder(tf.float32, shape=[None] + list(env.observation_space.shape))
        self._act = tf.placeholder(tf.float32, shape=[None] + list(env.action_space.shape))
        self._build_net(layers, lr)

    def _build_net(self, layers, lr):
        fc = slim.fully_connected(self._obs, layers[0], scope='fc1',
                                  activation_fn=tf.nn.relu)
        for i, dim in enumerate(layers[1:]):
            fc = slim.fully_connected(fc, dim, scope='fc%d' % (i + 2),
                                      activation_fn=tf.nn.relu)
        self.act = slim.fully_connected(fc, self._act_dim, scope='action',
                                        activation_fn=None)
        self.loss = tf.reduce_mean(tf.reduce_sum((self.act - self._act)**2, axis=1))
        self.train_op = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(self.loss)

    def train(self, obs, acts):
        sess = tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self._obs: obs, self._act: acts})
        return loss

    def action(self, ob):
        sess = tf.get_default_session()
        act = sess.run(self.act, feed_dict={self._obs: ob})
        return act

    def validate(self, env, max_steps):
        reward = 0.
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            action = self.action(obs[None, :])[0]
            obs, r, done, _ = env.step(action)
            reward += r
            steps += 1
            if steps >= max_steps or done:
                break

        return reward


def generating_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=2,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    generate_expert_data(args.envname, args.num_rollouts, render=args.render, save=True)


if __name__ == '__main__':
    generating_test()
