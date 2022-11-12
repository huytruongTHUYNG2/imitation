from typing import Optional

import torch as th
from stable_baselines3.common import base_class, vec_env
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets

from wgail_training import WGAN_AdversarialTrainer


class WGANRewardNet(reward_nets.RewardNet):
    def __init__(self, base: reward_nets.RewardNet):
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        outputs = self.base.forward(state, action, next_state, done)
        return outputs


class WGAIL(WGAN_AdversarialTrainer):
    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        **kwargs,
    ):
        # Raw self._reward_net is critic's score
        reward_net = reward_net.to(gen_algo.device)
        # Process it to produce output suitable for RL training
        self._processed_reward = WGANRewardNet(reward_net)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

    def score_expert_is_positive(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        del log_policy_act_prob
        score = self._reward_net(state, action, next_state, done)
        assert score.shape == state.shape[:1]
        return score

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._processed_reward

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._processed_reward