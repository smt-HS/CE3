import gym
import ds_gym
import argparse
from importlib import import_module
from baselines.common.cmd_util import parse_unknown_args
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
import os
import tensorflow as tf
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np
import sys

ppo_default_params = dict(
    nsteps=2048,
    nminibatches=32,
    lam=0.95,
    gamma=0.99,
    noptepochs=10,
    log_interval=1,
    ent_coef=0.0,
    lr=lambda f: 3e-4 * f,
    cliprange=0.2,
    value_network='copy'
)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='ds-v10')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--topic', help="Topic ID", default='dd17-1', type=str)
    parser.add_argument("--corpus", help="Directory BOW representation of corpus", type=str, default="corpus_bow")
    # parser.add_argument("--gpu_id", help="GPU for environment to use", type=int, default=0)
    parser.add_argument("--truth", help="Ground truth file of TREC DD", default='data/truth_data_nyt_2017_v2.3.xml',
                        type=str)
    return parser


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def make_env(env_id, env_type, topic, truth, corpus, subrank=0, seed=None, reward_scale=1.0, gamestate=None,
             wrapper_kwargs={}):
    mpi_rank = 0

    env = gym.make(env_id)
    env.set_topic(topic, truth_path=truth, corpus_path=corpus, env_rank=subrank)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    return env


def build_env(args, env_type, env_id):
    nenv = args.num_env or 1
    seed = args.seed
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    # env = make_vec_env(env_id, env_type, nenv, seed, reward_scale=args.reward_scale)
    set_global_seeds(seed)

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            topic=args.topic,
            truth=args.truth,
            corpus=args.corpus,
            subrank=rank,
            seed=seed,
            reward_scale=args.reward_scale,
            gamestate=None,
            wrapper_kwargs={}
        )

    env = DummyVecEnv([make_thunk(i) for i in range(nenv)])
    # env = SubprocVecEnv([make_thunk(i) for i in range(nenv)])

    return env


def build_network(board):
    h1 = tf.layers.conv2d(inputs=board, filters=8, kernel_size=[2, 2], kernel_initializer=tf.initializers.orthogonal,
                          activation=tf.nn.relu)
    h2 = tf.layers.conv2d(inputs=h1, filters=16, kernel_size=[2, 2], kernel_initializer=tf.initializers.orthogonal,
                          activation=tf.nn.relu)
    h3 = tf.layers.flatten(h2)

    h4 = tf.layers.dense(inputs=h3, units=32, activation=tf.nn.relu)

    return h4



def train(args, extra_args):
    env_type, env_id = 'ds', args.env
    print('env_type: {}'.format(env_type))
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = ppo_default_params
    alg_kwargs.update(extra_args)

    env = build_env(args, env_type, env_id)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        network=build_network,  # customized neural network
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def main(argv):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(argv)
    extra_args = parse_cmdline_kwargs(unknown_args)

    rank = 0
    logger.configure()

    model, env = train(args, extra_args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)

    """
    if args.play:
        logger.log("Running trained model")
        env = build_env(args, 'ds', args.env, play=True)
        obs = env.reset()

        def initialize_placeholders(nlstm=128, **kwargs):
            return np.zeros((args.num_env or 1, 2 * nlstm)), np.zeros((1))

        state, dones = initialize_placeholders(**extra_args)
        while True:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                # obs = env.reset()
                break

        env.close()
    """


if __name__ == "__main__":
    main(sys.argv[1:])
