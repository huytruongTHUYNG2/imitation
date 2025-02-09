import abc
import collections
import dataclasses
import logging
import os
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from imitation.algorithms.adversarial.common import compute_train_stats
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util


class WGAN_AdversarialTrainer(base.DemonstrationAlgorithm[types.Transitions]):

    venv: vec_env.VecEnv

    venv_train: vec_env.VecEnv

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        n_disc_updates_per_round: int = 5,  # default value in WGAN research paper
        log_dir: str = "output/",
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        gen_train_timesteps: Optional[int] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        allow_variable_horizon: bool = False,
    ):
        self.demo_batch_size = demo_batch_size
        self._demo_data_loader = None
        self._endless_expert_iterator = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo
        self._reward_net = reward_net.to(gen_algo.device)
        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = self._disc_opt_cls(
            self._reward_net.parameters(),
            **self._disc_opt_kwargs,
        )

        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        venv = self.venv_buffering = wrappers.BufferingWrapper(self.venv)

        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_wrapped = venv
            self.gen_callback = None
        else:
            venv = self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
                venv,
                reward_fn=self.reward_train.predict_processed,
            )
            self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        if gen_train_timesteps is None:
            gen_algo_env = self.gen_algo.get_env()
            assert gen_algo_env is not None
            self.gen_train_timesteps = gen_algo_env.num_envs
            if hasattr(self.gen_algo, "n_steps"):  # on policy
                self.gen_train_timesteps *= self.gen_algo.n_steps
        else:
            self.gen_train_timesteps = gen_train_timesteps

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_train_timesteps
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity,
            self.venv,
        )

    @property
    def policy(self) -> policies.BasePolicy:
        return self.gen_algo.policy

    @abc.abstractmethod
    def score_expert_is_positive(
        self,
        state_action_next_state_done: th.Tensor,
        padding: tuple
    ) -> th.Tensor:
        """ Returns the score for expert trajectories and score for policy trajectories """

    @property
    @abc.abstractmethod
    def reward_train(self) -> reward_nets.RewardNet:
        """Reward used to train generator policy."""

    @property
    @abc.abstractmethod
    def reward_test(self) -> reward_nets.RewardNet:
        """Reward used to train policy at "test" time after adversarial training."""

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Optional[Mapping[str, float]]:
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            batch = self._make_disc_train_batch(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            s, a, ns, d = batch["state"], batch["action"], batch["next_state"], batch["done"]
            # flip sign because loss minimization
            grad = -batch["score_expert_is_positive"].float()

            d = d[:, None]  # add extra dimension so we can pad the 2nd dimension
            max_pad = max(s.shape[1], a.shape[1], ns.shape[1], d.shape[1])
            pad = (max_pad - s.shape[1], max_pad - a.shape[1], max_pad - ns.shape[1], max_pad - d.shape[1])
            # add padding so we can stack
            s = F.pad(s, (0, pad[0]))
            a = F.pad(a, (0, pad[1]))
            ns = F.pad(ns, (0, pad[2]))
            d = F.pad(d, (0, pad[3]))
            # get score
            disc_score = self.score_expert_is_positive(torch.vstack((s, a, ns, d)), pad)

            # find the dividing index between expert data and learner data
            ind = int(disc_score.shape[0] / 2)
            # perform brute-force linear-interpolation of real and generated data
            eps = torch.rand(ind, 1)
            mix_s = eps * s[:ind] + (1 - eps) * s[ind:]
            mix_a = eps * a[:ind] + (1 - eps) * a[ind:]
            mix_ns = eps * ns[:ind] + (1 - eps) * ns[ind:]
            mix_d = eps * d[:ind] + (1 - eps) * d[ind:]
            # the mixed inputs are still valid padded
            mix_inp = torch.tensor(torch.vstack((mix_s, mix_a, mix_ns, mix_d)), requires_grad=True)
            # get score
            disc_score_mix = self.score_expert_is_positive(mix_inp, pad)

            # create the losses for back-propagation
            normal_loss = disc_score
            gradient_penalty_loss = torch.mean(
                (torch.autograd.grad(
                    outputs=disc_score_mix,
                    inputs=mix_inp,
                    grad_outputs=torch.ones_like(disc_score_mix),
                    create_graph=True,
                    retain_graph=True
                )[0]
                 .view(disc_score_mix.shape[0], -1)
                 .norm(2, 1) - 1
                ) ** 2
            )

            # do gradient step
            self._disc_opt.zero_grad()
            # loss for expert data
            normal_loss[:ind].backward(gradient=grad[:ind], retain_graph=True)
            # loss for learner data
            normal_loss[ind:].backward(gradient=grad[ind:], retain_graph=True)
            # gradient penalty loss
            gradient_penalty_loss.backward()

            self._disc_opt.step()
            self._disc_step += 1

            # clamp discriminator parameters to a range [-c, c] for stability
            c = 0.01  # default value in WGAN research paper
            for p in self._reward_net.parameters():
                p.data.clamp_(-c, c)

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_score,
                    batch["score_expert_is_positive"],
                    normal_loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_score", disc_score.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}

        with self.logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                **learn_kwargs,
            )
            self._global_step += 1

        gen_trajs, ep_lens = self.venv_buffering.pop_trajectories()
        self._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_train_timesteps)
            for _ in range(self.n_disc_updates_per_round):
                with networks.training(self.reward_train):
                    # switch to training mode (affects dropout, normalization)
                    self.train_disc()
            if callback:
                callback(r)
            self.logger.dump(self._global_step)

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.reward_train.device)

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
    ) -> Optional[th.Tensor]:
        if isinstance(self.policy, policies.ActorCriticPolicy):
            # policies.ActorCriticPolicy has a concrete implementation of
            # evaluate_actions to generate log_policy_act_prob given obs and actions.
            _, log_policy_act_prob_th, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )
        elif isinstance(self.policy, sac_policies.SACPolicy):
            gen_algo_actor = self.policy.actor
            assert gen_algo_actor is not None
            # generate log_policy_act_prob from SAC actor.
            mean_actions, log_std, _ = gen_algo_actor.get_action_dist_params(obs_th)
            distribution = gen_algo_actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
            )
            # SAC applies a squashing function to bound the actions to a finite range
            # `acts_th` need to be scaled accordingly before computing log prob.
            # Scale actions only if the policy squashes outputs.
            assert self.policy.squash_output
            scaled_acts_th = self.policy.scale_action(acts_th)
            log_policy_act_prob_th = distribution.log_prob(scaled_acts_th)
        else:
            return None
        return log_policy_act_prob_th

    def _make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Mapping[str, th.Tensor]:
        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            gen_samples = self._gen_replay_buffer.sample(self.demo_batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.demo_batch_size):
            raise ValueError(
                "Need to have exactly self.demo_batch_size number of expert and "
                "generator samples, each. "
                f"(n_gen={n_gen} n_expert={n_expert} "
                f"demo_batch_size={self.demo_batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        n_samples = n_expert + n_gen
        assert n_expert == len(expert_samples["acts"])
        assert n_expert == len(expert_samples["next_obs"])
        assert n_gen == len(gen_samples["acts"])
        assert n_gen == len(gen_samples["next_obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        # notice that the labels use the convention that expert samples are
        # positively scored while generator is negatively scored
        score_expert_is_positive = np.concatenate(
            [np.ones(n_expert, dtype=int), -np.ones(n_gen, dtype=int)],
        )

        # Calculate generator-policy log probabilities.
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            acts_th = th.as_tensor(acts, device=self.gen_algo.device)
            log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
            if log_policy_act_prob is not None:
                assert len(log_policy_act_prob) == n_samples
                log_policy_act_prob = log_policy_act_prob.reshape((n_samples,))
            del obs_th, acts_th  # unneeded

        obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
            obs,
            acts,
            next_obs,
            dones,
        )
        batch_dict = {
            "state": obs_th,
            "action": acts_th,
            "next_state": next_obs_th,
            "done": dones_th,
            "score_expert_is_positive": self._torchify_array(score_expert_is_positive),
            "log_policy_act_prob": log_policy_act_prob,
        }

        return batch_dict