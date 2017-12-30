"""
Some of the codes adapted from hiwonjoon's repo
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import smooth, generate_expert_data, test_run, Policy
import pickle
import gym
from gym import wrappers


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--expert_policy_file', dest='expert_policy_file',
                        type=str, default=None)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epoch', dest='epoch', type=int, default=100)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    return args


def behaviral_cloning(save_fig=True, save_env=False):
    if sys.platform == 'darwin':
        # mac
        matplotlib.use('TkAgg')
        print('Using Mac OS')
    args = parse_arg()
    bs = args.batch_size
    expert_data = 'output/' + args.envname + '_expert_' + str(args.num_rollouts) + '.pkl'
    if os.path.exists(expert_data):
        with open(expert_data, 'rb') as f:
            expert_data = pickle.loads(f.read())
    else:
        expert_data = generate_expert_data(args.envname, args.num_rollouts,
                                           expert_policy_file=args.expert_policy_file,
                                           max_timesteps=args.max_timesteps,
                                           render=False, save=True)

    obs = expert_data['observations']
    # squeeze [N, 1, dim] to [N, dim]
    acts = np.squeeze(expert_data['actions'])
    num_train = obs.shape[0]
    print('Number of training examples: %d' % num_train)

    # set up env
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    policy = Policy(env, [64, 64, 64], args.lr)

    # set up session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        for epoch in tqdm(range(args.epoch)):
            perm = np.random.permutation(num_train)
            obs = obs[perm]
            acts = acts[perm]

            loss = 0.
            for i in range(0, num_train, bs):
                loss += policy.train(obs[i: i + bs], acts[i: i + bs])

            losses.append(loss / num_train)
            reward = policy.validate(env, max_steps)
            tqdm.write("Epoch %3d Loss %f Reward %.2f" % (epoch, loss / num_train, reward))

        # draw traing curve
        if save_fig:
            assert os.path.exists('./output')
            plt.plot(smooth(losses, 0))
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title('bc training with lr-%s' % args.lr)
            plt.savefig('./output/bc_loss_%s.pdf' % args.lr, format='pdf')

        if save_env:
            env = wrappers.Monitor(env, './output/' + args.envname, force=True)

        results = test_run(env, 5, policy.action, max_steps, args.render)
        returns = results['returns']
        print('returns:', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    behaviral_cloning()
