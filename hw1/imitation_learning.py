"""
Some of the codes adapted from hiwonjoon's repo
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import gym
from gym import wrappers
import load_policy
from utils import smooth, generate_expert_data, test_run, Policy
from matplotlib.backends.backend_pdf import PdfPages


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--expert_policy_file', dest='expert_policy_file',
                        type=str, default=None)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epoch', dest='epoch', type=int, default=100)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--save_env', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    return args


def imitation_learning():
    args = parse_arg()
    bs = args.batch_size
    expert_data = 'output/' + args.envname + '_expert_' + str(args.num_rollouts) + '.pkl'
    if os.path.exists(expert_data):
        with open(expert_data, 'rb') as f:
            expert_data = pickle.loads(f.read())
        print('expert - mean return', np.mean(expert_data['returns']))
        print("expert - std of return", np.std(expert_data['returns']))
    else:
        expert_data = generate_expert_data(args.envname, args.num_rollouts,
                                           expert_policy_file=args.expert_policy_file,
                                           max_timesteps=args.max_timesteps,
                                           render=False, save=True)

    if args.dagger:
        print('loading and building expert policy')
        expert_policy_file = args.expert_policy_file or 'experts/' + \
            args.envname + '.pkl'
        expert_policy = load_policy.load_policy(expert_policy_file)
        print('loaded and built')

    obs = expert_data['observations']
    # squeeze [N, 1, dim] to [N, dim]
    acts = np.squeeze(expert_data['actions'])
    print('Number of training examples: %d' % obs.shape[0])

    # set up env
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    policy = Policy(env, [64, 64, 64], args.lr)

    # set up session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    returns = []
    with tf.Session(config=tfconfig) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        losses = []
        for epoch in tqdm(range(args.epoch)):
            num_train = obs.shape[0]
            perm = np.random.permutation(num_train)
            obs = obs[perm]
            acts = acts[perm]

            loss = 0.
            for i in range(0, num_train, bs):
                loss += policy.train(obs[i: i + bs], acts[i: i + bs])

            losses.append(loss / num_train)
            result = policy.validate(env, max_steps)
            tqdm.write("Epoch %3d Loss %f Reward %.2f" % (epoch, loss / num_train, result['reward']))

            if args.dagger:
                obs = np.concatenate([obs, result['observations']], axis=0)
                expert_acts = []
                for ob in result['observations']:
                    expert_acts.append(expert_policy(ob[None, :]))
                squeezed_acts = np.squeeze(np.array(expert_acts), axis=[1])
                acts = np.concatenate([acts, squeezed_acts], axis=0)

                assert obs.shape[0] == acts.shape[0]
                tqdm.write('update training examples with: %d' % obs.shape[0])
                results = test_run(env, args.num_rollouts, policy.action, max_steps, False)
                returns.append([np.mean(results['returns']),
                                np.std(results['returns'])])

        # save ckpt
        save_path = saver.save(sess, "output/" + args.envname + '_' +
                               str(args.num_rollouts) + '.ckpt')
        print("Model saved in file: %s" % save_path)
        # saver.restore(sess, "output/" + args.envname + '_' +
        #                        str(args.num_rollouts) + '.ckpt')

        # draw traing curve
        if args.save_fig:
            assert os.path.exists('./output')
            plt.plot(smooth(losses, 0))
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title('bc training with lr-%s' % args.lr)
            plt.savefig('./output/bc_loss_%s.pdf' % args.lr, format='pdf')

        if args.save_env:
            env = wrappers.Monitor(env, './output/' + args.envname, force=True)

        if args.dagger:
            # save returns of every epoch
            assert os.path.exists('./output')
            with open('output/dagger_' + args.envname + '.pkl', 'wb') as f:
                pickle.dump(returns, f, pickle.HIGHEST_PROTOCOL)

        results = test_run(env, args.num_rollouts, policy.action, max_steps, args.render)
        print('returns:', results['returns'])
        print('mean return', np.mean(results['returns']))
        print('std of return', np.std(results['returns']))


def dagger_plot():
    file = 'output/dagger_Humanoid-v1.pkl'
    with open(file, 'rb') as f:
        results = pickle.loads(f.read())

    means = [i[0] for i in results]
    stds = [i[1] for i in results]

    iters = list(range(len(results)))
    dagger = plt.figure()
    dag, = plt.plot(iters, means, label='DAgger')
    plt.errorbar(iters, means, yerr=stds, fmt='o')
    plt.title('DAgger Iters vs Rewards', fontsize=20)
    plt.xlabel('Iters')
    plt.ylabel('Rewards')
    expert = plt.axhline(y=10399.5657012, color='k', label='Expert')
    bc = plt.axhline(y=1440.88102146, color='r', label='Behavioral Cloning')
    plt.legend(loc=4)
    pp = PdfPages('dagger_plot.pdf')
    pp.savefig(dagger)
    pp.close()


if __name__ == '__main__':
    imitation_learning()
    # dagger_plot()
