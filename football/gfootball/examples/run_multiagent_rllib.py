from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import argparse
import numpy as np
import gfootball.env as football_env
import gymnasium as gym
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=3)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')


class RllibGFootball(MultiAgentEnv):
    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.env = football_env.create_environment(
            env_name='test_example_multiagent',
            stacked=False,
            logdir=os.path.join(tempfile.gettempdir(), 'rllib_test'),
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            dump_frequency=0,
            representation='simple115v2',
            number_of_left_players_agent_controls=num_agents)
        
        self._agent_ids = set([f'agent_{i}' for i in range(num_agents)])
        
        # Get single agent spaces
        if hasattr(self.env.action_space, 'nvec'):
            self._action_space = spaces.Discrete(int(self.env.action_space.nvec[0]))
        else:
            self._action_space = spaces.Discrete(self.env.action_space.n)
        
        obs_shape = self.env.observation_space.shape
        if len(obs_shape) > 1:
            single_obs_shape = obs_shape[1:]
        else:
            single_obs_shape = obs_shape
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=single_obs_shape, dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            agent_id: self._observation_space for agent_id in self._agent_ids
        })
        self.action_space = spaces.Dict({
            agent_id: self._action_space for agent_id in self._agent_ids
        })

    def reset(self, *, seed=None, options=None):
        original_obs = self.env.reset()
        obs = {}
        infos = {}
        for x in range(self.num_agents):
            agent_id = f'agent_{x}'
            if self.num_agents > 1:
                obs[agent_id] = np.array(original_obs[x], dtype=np.float32)
            else:
                obs[agent_id] = np.array(original_obs, dtype=np.float32)
            infos[agent_id] = {}
        return obs, infos

    def step(self, action_dict):
        # Build actions list in order
        actions = []
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            if agent_id in action_dict:
                actions.append(action_dict[agent_id])
            else:
                actions.append(0)  # Default action
        
        if self.num_agents == 1:
            actions = actions[0]
        
        o, r, d, i = self.env.step(actions)
        
        rewards = {}
        obs = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        for x in range(self.num_agents):
            agent_id = f'agent_{x}'
            if self.num_agents > 1:
                rewards[agent_id] = float(r[x])
                obs[agent_id] = np.array(o[x], dtype=np.float32)
            else:
                rewards[agent_id] = float(r)
                obs[agent_id] = np.array(o, dtype=np.float32)
            terminateds[agent_id] = d
            truncateds[agent_id] = False
            infos[agent_id] = i
        
        terminateds['__all__'] = d
        truncateds['__all__'] = False
        
        return obs, rewards, terminateds, truncateds, infos


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=0)

    register_env('gfootball', lambda _: RllibGFootball(args.num_agents))
    single_env = RllibGFootball(args.num_agents)
    obs_space = single_env._observation_space
    act_space = single_env._action_space

    def gen_policy(_):
        return (None, obs_space, act_space, {})

    policies = {
        'policy_{}'.format(i): gen_policy(i) for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    tune.run(
        'PPO',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=50,
        config={
            'env': 'gfootball',
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'vf_clip_param': 10.0,
            'entropy_coeff': 0.01,
            'train_batch_size': 2000,
            'sgd_minibatch_size': 500,
            'num_sgd_iter': 10,
            'num_workers': 0,
            'num_envs_per_worker': 1,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'vf_share_layers': True,
            'num_gpus': 0,
            'num_gpus_per_worker': 0,
            'lr': 2.5e-4,
            'log_level': 'WARN',
            'simple_optimizer': args.simple,
            'framework': 'torch',
            'disable_env_checking': True,
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': lambda agent_id, episode=None, worker=None, **kwargs: policy_ids[int(agent_id.split('_')[1]) % len(policy_ids)],
            },
        },
    )
