# **********************************
# code adapted from Ted Xiao's repo
# **********************************
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Reshape


def smooth(a, beta=0.8):
    '''smooth the curve'''
    for i in range(1, len(a)):
        a[i] = beta * a[i - 1] + (1 - beta) * a[i]
    return a


def main():
    if sys.platform == 'darwin':
        # mac
        matplotlib.use('TkAgg')
        print('Using Mac OS')
    # get expert_data
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epoch', dest='epoch', type=int, default=100)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    with open(args.expert_data, 'rb') as f:
        expert_data = pickle.loads(f.read())

    obs = expert_data['observations']
    acts = expert_data['actions']
    acts = np.squeeze(acts, axis=[1])
    num_exmple = obs.shape[0]

    print('number of training examples: ', num_exmple)
    print('dimension of observation: ', obs[0].shape)
    print('dimension of action: ', acts[0].shape)

    # shuffle_list = np.arange(num_exmple)
    # np.random.shuffle(shuffle_list)
    # obs, acts = obs[shuffle_list], acts[shuffle_list]
    obs, acts = shuffle(obs, acts, random_state=0)
    split = int(0.8 * num_exmple)
    obs_train, acts_train = obs[:split], acts[:split]
    obs_val, acts_val = obs[split:], acts[split:]

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(obs.shape[1],)))
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(acts.shape[1], activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(obs_train, acts_train, batch_size=args.batch_size, epochs=args.epoch, verbose=1)
    score = model.evaluate(obs_val, acts_val, verbose=1)

    model.save('output/' + args.envname + '_bc.h5')

    # set up session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    num_train = obs_train.shape[0]
    shuffle_list = np.arange(num_train)
    losses = []
    with tf.Session(config=tfconfig) as sess:
        # model = Behavioral_clone(obs.shape[1], acts.shape[1])
        # model.build_net([128, 256, 512, 256, 128], lr=args.lr)
        # sess.run(tf.global_variables_initializer())

        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        model = load_model('output/' + args.envname + '_bc.h5')
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # action = model.action(sess, obs[None, :])
                obs = obs.reshape(1, len(obs))
                action = (model.predict(obs, batch_size=64, verbose=0))
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


if __name__ == '__main__':
    main()
