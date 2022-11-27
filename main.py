import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

from wgail import WGAIL

import seals

rng = np.random.default_rng(0)

# make demonstrations by using a PPO expert ############################################################################
env = gym.make("seals/CartPole-v0")
expert = PPO(policy=MlpPolicy, env=env, n_steps=64, seed=2)
expert.learn(total_timesteps=1000)
# this is the demonstrations
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/CartPole-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

# GAIL #################################################################################################################
venv = make_vec_env("seals/CartPole-v0", n_envs=8)
learner = PPO(env=venv, policy=MlpPolicy)
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(total_timesteps=50_000)
gail_rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)


# WGAIL ################################################################################################################
venv = make_vec_env("seals/CartPole-v0", n_envs=8)
learner = PPO(env=venv, policy=MlpPolicy)
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)

wgail_trainer = WGAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=20,  # update critic 20 times per training round is best for WGAIL
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

wgail_trainer.train(total_timesteps=50_000)
wgail_rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)

# comparison ###########################################################################################################
print("GAIL Rewards:", gail_rewards)
print("WGAIL Rewards:", wgail_rewards)


