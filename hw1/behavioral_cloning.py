import pickle
import tensorflow as tf
import numpy as np
import pprint
import tf_util
import gym
# import load_policy
import argparse
# import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt


def smooth(a, beta=0.8):
    '''smooth the curve'''
    for i in range(1, len(a)):
        a[i] = beta * a[i - 1] + (1 - beta) * a[i]
    return a


class Behavioral_clone(object):
    """docstring for Behavioral_clone"""

    def __init__(self, in_dim, out_dim):
        super(Behavioral_clone, self).__init__()
        self.out_dim = out_dim
        self.obs = tf.placeholder(tf.float32, shape=[None, in_dim])
        self.act_gt = tf.placeholder(tf.float32, shape=[None, out_dim])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])

    def build_net(self, net_param, lr=1e-3, weight_decay=None, use_dropout=False):
        if weight_decay:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = tf.no_regularizer

        fc = tf.contrib.layers.fully_connected(self.obs, net_param[0],
                                               weights_regularizer=regularizer)
        if use_dropout:
            fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)

        for dim in net_param[1:]:
            fc = tf.contrib.layers.fully_connected(fc, dim,
                                                   weights_regularizer=regularizer)
            if use_dropout:
                fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)

        self.act = tf.contrib.layers.fully_connected(fc, self.out_dim, activation_fn=None)
        # self.loss = tf.nn.l2_loss(self.act_gt - self.act)
        self.loss = tf.reduce_mean(tf.reduce_sum((self.act_gt - self.act)**2, axis=1))
        # may add exponentially decay
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(lr, global_step, 1000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess, obs, act_gt, keep_prob=1.):
        feed = {self.obs: obs, self.act_gt: act_gt,
                self.keep_prob: keep_prob}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)

        return loss

    def action(self, sess, ob):
        feed = {self.obs: ob, self.keep_prob: 1.}
        act = sess.run(self.act, feed_dict=feed)

        return act

    def evaluate(self, sess, obs, acts):
        feed = {self.obs: obs, self.act_gt: acts,
                self.keep_prob: 1.}
        loss = sess.run(self.loss, feed_dict=feed)
        return loss


def main():
    # get expert_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', dest='envname', type=str, default='Hopper-v1')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--epoch', dest='epoch', type=int, default=10)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    with open('./output/expert_data_' + args.envname + '.pkl', 'rb') as f:
        expert_data = pickle.loads(f.read())

    obs = expert_data['observations']
    acts = expert_data['actions']
    acts = np.squeeze(acts, axis=[1])
    num_exmple = obs.shape[0]

    print('number of training examples: ', num_exmple)
    print('dimension of observation: ', obs[0].shape)
    print('dimension of action: ', acts[0].shape)

    shuffle_list = np.arange(num_exmple)
    np.random.shuffle(shuffle_list)
    obs, acts = obs[shuffle_list], acts[shuffle_list]
    split = int(0.8 * num_exmple)
    obs_train, acts_train = obs[:split], acts[:split]
    obs_val, acts_val = obs[split:], acts[split:]

    # set up session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    num_train = obs_train.shape[0]
    shuffle_list = np.arange(num_train)
    bs = args.batch_size
    losses = []
    with tf.Session(config=tfconfig) as sess:
        model = Behavioral_clone(obs.shape[1], acts.shape[1])
        model.build_net([128, 128, 128], lr=args.lr)
        sess.run(tf.global_variables_initializer())
        for e in range(args.epoch):
            # do random shuffle
            np.random.shuffle(shuffle_list)
            obs_train = obs_train[shuffle_list]
            acts_train = acts_train[shuffle_list]
            for i in range(num_train // bs):
                ob_batch = obs_train[i * bs: (i + 1) * bs]
                act_batch = acts_train[i * bs: (i + 1) * bs]
                loss = model.train(sess, ob_batch, act_batch)
                if i % 100 == 0:
                    losses.append(loss)
                    print("loss: %.4f" % loss)

        print("validation loss: {}".format(model.evaluate(sess, obs_val, acts_val)))

        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.action(sess, obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
    # drop plot
    plt.plot(smooth(losses))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('training process %s ' % args.lr)
    # plt.show()
    plt.savefig('./output/bc_loss_%s.pdf' % args.lr, format='pdf')


if __name__ == '__main__':
    main()
