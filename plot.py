import numpy as np
import seals
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
import imitation.util.networks
from imitation.util.util import make_vec_env
import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import matplotlib.pyplot as plt
from wgail import WGAIL

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
rng = np.random.default_rng(0)

env = gym.make("seals/CartPole-v0")
expert = PPO(policy=MlpPolicy, env=env, seed=2, batch_size=64,
             ent_coef=0.0, learning_rate=0.0003, n_epochs=10, n_steps=64)
expert.learn(1000)

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/CartPole-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

venv = make_vec_env("seals/CartPole-v0", n_envs=8)
gail_learner = PPO(env=venv, policy=MlpPolicy, batch_size=64, ent_coef=0.0, learning_rate=0.0003, n_epochs=10, )
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)

# GAIL########################################################################

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=gail_learner,
    reward_net=reward_net,
)

expert_rewards, _ = evaluate_policy(
    expert, venv, 100, return_episode_rewards=True)

gail_trainer.train(20000)

gail_learner_rewards, _ = evaluate_policy(
    gail_learner, venv, 100, return_episode_rewards=True
)

# WGAIL ################################################################################################################
venv = make_vec_env("seals/CartPole-v0", n_envs=8)
wgail_learner = PPO(env=venv, policy=MlpPolicy)
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
    gen_algo=wgail_learner,
    reward_net=reward_net,
)

wgail_trainer.train(total_timesteps= 20000)
wgail_learner_rewards, _ = evaluate_policy(wgail_learner, venv, 100, return_episode_rewards=True)

# comparison ###########################################################################################################

# print the mean and plot the results
print(np.mean(expert_rewards))
print(np.mean(gail_learner_rewards))
print(np.mean(wgail_learner_rewards))

plt.hist(
    [expert_rewards, gail_learner_rewards, wgail_learner_rewards],
    label=["expert", "gail", 'wgail'],
)
plt.title('expert, gail and wgail rewards')
plt.legend()
plt.xlabel('Rewards')
plt.ylabel('Frequency')
plt.show()

rewards, _ = evaluate_policy(expert, venv, 100, return_episode_rewards=True)
print("Rewards_expert:", rewards)
rewards, _ = evaluate_policy(gail_learner, venv, 100, return_episode_rewards=True)
print("Rewards_gail:", rewards)
rewards, _ = evaluate_policy(wgail_learner, venv, 100, return_episode_rewards=True)
print("Rewards_wgail:", rewards)

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


def show_videos(video_path='', prefix=''):
    """
  Taken from https://github.com/eleurent/highway-env

  :param video_path: (str) Path to the folder containing videos
  :param prefix: (str) Filter the video, showing only the only starting with this prefix
  """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


record_video('seals/CartPole-v0', expert, video_length=3000, prefix='expert-cartpole')
record_video('seals/CartPole-v0', gail_learner, video_length=3000, prefix='gail-learner-cartpole')
record_video('seals/CartPole-v0', wgail_learner, video_length=3000, prefix='wgail-learner-cartpole')

