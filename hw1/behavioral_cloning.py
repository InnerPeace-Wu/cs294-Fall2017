import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import argparse
import tensorflow.contrib.slim as slim
# import matplotlib.pyplot as plt


def smooth(a, beta=0.8):
    '''smooth the curve'''

    for i in xrange(1, len(a)):
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

    def build_net(self, net_param, lr=5e-3, weight_decay=None, use_dropout=False):
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

        self.act = tf.contrib.layers.fully_connected(fc, self.out_dim)
        self.loss = tf.nn.l2_loss(self.act_gt - self.act)
        # may add exponentially decay
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


def main():
    # get expert_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', dest='envname', type=str, default='Hopper-v1')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('--epoch', dest='epoch', type=int, default=10)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-2)
    args = parser.parse_args()

    with open('./expert_data_' + args.envname + '.pkl', 'rb') as f:
        expert_data = pickle.loads(f.read())

    obs = expert_data['observations']
    acts = expert_data['actions']
    acts = np.squeeze(acts, axis=[1])
    num_exmple = obs.shape[0]
    shuffle_list = np.arange(num_exmple)
    bs = args.batch_size

    print('number of training examples: ', num_exmple)
    print('dimension of observation: ', obs[0].shape)
    print('dimension of action: ', acts[0].shape)

    # set up session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    losses = []
    with tf.Session(config=tfconfig) as sess:
        model = Behavioral_clone(obs[0].shape[0], acts[0].shape[0])
        model.build_net([40, 40, 40], lr=args.lr)
        sess.run(tf.global_variables_initializer())
        for e in range(args.epoch):
            # do random shuffle
            np.random.shuffle(shuffle_list)
            obs = obs[shuffle_list]
            acts = acts[shuffle_list]
            for i in range(num_exmple // bs):
                ob_batch = obs[i * bs: (i + 1) * bs]
                act_batch = acts[i * bs: (i + 1) * bs]
                loss = model.train(sess, ob_batch, act_batch)
                losses.append(loss)
                print("loss: %.4f" % loss)

    # drop plot


if __name__ == '__main__':
    main()
