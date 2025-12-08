from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import argparse
import numpy as np
import gfootball.env as football_env
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

# Argument parser

parser = argparse.ArgumentParser()
parser.add_argument('--num-iters', type=int, default=10000)
parser.add_argument('--checkpoint-freq', type=int, default=100)
parser.add_argument('--self-play', action='store_true', help='Enable self-play training')
parser.add_argument('--simple', action='store_true')

# Multi-Agent Environment with Defensive Reward Shaping
class DefensiveGFootball5v5(MultiAgentEnv):
    """
    5v5 Multi-Agent Environment with Defensive Reward Shaping.
    
    Reward Structure - Defensive Focus:
    - Goal conceded: -10.0 (massive penalty)
    - Goal scored: +1.0 (small reward, not the focus)
    - Clean sheet bonus: +5.0 per episode without conceding
    - Ball interception: +0.5
    - Possession in defensive third: +0.1 (encourage active defense)
    """
    
    def __init__(self, num_left_agents=5, num_right_agents=0):
        super().__init__()
        self.num_left_agents = num_left_agents
        self.num_right_agents = num_right_agents
        self._num_agents = num_left_agents + num_right_agents
        
        # Create environment 
        self.env = football_env.create_environment(
            env_name='5_vs_5',
            stacked=False,  
            representation='simple115v2',
            logdir=os.path.join(tempfile.gettempdir(), 'defensive_5v5'),
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            number_of_left_players_agent_controls=num_left_agents,
            number_of_right_players_agent_controls=num_right_agents)
        
        # Agent IDs
        self._agent_ids = set()
        for i in range(num_left_agents):
            self._agent_ids.add(f'left_{i}')
        for i in range(num_right_agents):
            self._agent_ids.add(f'right_{i}')
        
        # Action space
        if hasattr(self.env.action_space, 'nvec'):
            self._action_space = spaces.Discrete(int(self.env.action_space.nvec[0]))
        else:
            self._action_space = spaces.Discrete(self.env.action_space.n)
        
        # Get the actual observation shape from a reset
        test_obs = self.env.reset()
        if self._num_agents > 1:
            single_obs = test_obs[0]
        else:
            single_obs = test_obs
        
        # Use the actual observation shape
        self.obs_shape = single_obs.shape
        print(f"Detected observation shape: {self.obs_shape}")
        
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            agent_id: self._observation_space for agent_id in self._agent_ids
        })
        self.action_space = spaces.Dict({
            agent_id: self._action_space for agent_id in self._agent_ids
        })
        
        # Tracking for defensive rewards
        self.prev_ball_owned_team = None
        self.prev_ball_position = None
        self.goals_conceded = 0
        self.goals_scored = 0
        self.episode_steps = 0
        self.prev_score = (0, 0)

    def _get_defensive_rewards(self, obs_raw, base_reward, done, info):
        """Calculate defensive-focused rewards for each agent."""
        rewards = {}
        
        try:
            if isinstance(obs_raw, (list, np.ndarray)):
                if self._num_agents > 1:
                    obs = obs_raw[0] if len(obs_raw) > 0 else obs_raw
                else:
                    obs = obs_raw
                
                ball_x = obs[0] if len(obs) > 0 else 0
                ball_owned_team = info.get('ball_owned_team', -1)
                
                score_diff = 0
                if 'score' in info:
                    left_score, right_score = info['score']
                    left_diff = left_score - self.prev_score[0]
                    right_diff = right_score - self.prev_score[1]
                    score_diff = left_diff - right_diff
                    self.prev_score = (left_score, right_score)
        except Exception:
            ball_x = 0
            ball_owned_team = -1
            score_diff = 0
        
        for agent_id in self._agent_ids:
            reward = 0.0
            is_left_team = agent_id.startswith('left_')
            
            # Goal conceded penalty
            if is_left_team and score_diff < 0:
                reward -= 10.0
                self.goals_conceded += 1
            elif not is_left_team and score_diff > 0:
                reward -= 10.0
            
            # Goal scored (small reward)
            if is_left_team and score_diff > 0:
                reward += 1.0
                self.goals_scored += 1
            elif not is_left_team and score_diff < 0:
                reward += 1.0
            
            # Ball interception reward
            if self.prev_ball_owned_team is not None:
                if is_left_team:
                    if self.prev_ball_owned_team == 1 and ball_owned_team == 0:
                        reward += 0.5
                else:
                    if self.prev_ball_owned_team == 0 and ball_owned_team == 1:
                        reward += 0.5
            
            # Possession in defensive third
            if is_left_team and ball_owned_team == 0 and ball_x > -0.1:
                reward += 0.05 
            else:
                if ball_owned_team == 1 and ball_x > 0.3:
                    reward += 0.1
            
            # Clean sheet bonus
            if done:
                if is_left_team and self.goals_conceded == 0:
                    reward += 5.0
                elif not is_left_team and self.goals_scored == 0:
                    reward += 5.0
            
            reward -= 0.001
            rewards[agent_id] = reward
        
        self.prev_ball_owned_team = ball_owned_team
        self.prev_ball_position = ball_x
        self.episode_steps += 1
        
        return rewards

    def reset(self, *, seed=None, options=None):
        original_obs = self.env.reset()
        
        self.prev_ball_owned_team = None
        self.prev_ball_position = None
        self.prev_score = (0, 0)
        self.goals_conceded = 0
        self.goals_scored = 0
        self.episode_steps = 0
        
        obs = {}
        infos = {}
        
        idx = 0
        for i in range(self.num_left_agents):
            agent_id = f'left_{i}'
            if self._num_agents > 1:
                obs[agent_id] = np.array(original_obs[idx], dtype=np.float32)
            else:
                obs[agent_id] = np.array(original_obs, dtype=np.float32)
            infos[agent_id] = {}
            idx += 1
        
        for i in range(self.num_right_agents):
            agent_id = f'right_{i}'
            if self._num_agents > 1:
                obs[agent_id] = np.array(original_obs[idx], dtype=np.float32)
            else:
                obs[agent_id] = np.array(original_obs, dtype=np.float32)
            infos[agent_id] = {}
            idx += 1
        
        return obs, infos

    def step(self, action_dict):
        actions = []
        for i in range(self.num_left_agents):
            agent_id = f'left_{i}'
            actions.append(action_dict.get(agent_id, 0))
        for i in range(self.num_right_agents):
            agent_id = f'right_{i}'
            actions.append(action_dict.get(agent_id, 0))
        
        if len(actions) == 1:
            actions = actions[0]
        
        o, r, d, info = self.env.step(actions)
        
        defensive_rewards = self._get_defensive_rewards(o, r, d, info)
        
        obs = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        idx = 0
        for i in range(self.num_left_agents):
            agent_id = f'left_{i}'
            if self._num_agents > 1:
                obs[agent_id] = np.array(o[idx], dtype=np.float32)
            else:
                obs[agent_id] = np.array(o, dtype=np.float32)
            terminateds[agent_id] = d
            truncateds[agent_id] = False
            infos[agent_id] = info
            idx += 1
        
        for i in range(self.num_right_agents):
            agent_id = f'right_{i}'
            if self._num_agents > 1:
                obs[agent_id] = np.array(o[idx], dtype=np.float32)
            else:
                obs[agent_id] = np.array(o, dtype=np.float32)
            terminateds[agent_id] = d
            truncateds[agent_id] = False
            infos[agent_id] = info
            idx += 1
        
        terminateds['__all__'] = d
        truncateds['__all__'] = False
        
        return obs, defensive_rewards, terminateds, truncateds, infos


class SelfPlayGFootball5v5(DefensiveGFootball5v5):
    """5v5 with Self-Play: Left team vs Right team."""
    def __init__(self):
        super().__init__(num_left_agents=5, num_right_agents=5)


# Global variable to store detected obs shape
DETECTED_OBS_SHAPE = None


def create_defensive_env(env_config):
    global DETECTED_OBS_SHAPE
    env = DefensiveGFootball5v5(num_left_agents=5, num_right_agents=0)
    DETECTED_OBS_SHAPE = env.obs_shape
    return env


def create_selfplay_env(env_config):
    global DETECTED_OBS_SHAPE
    env = SelfPlayGFootball5v5()
    DETECTED_OBS_SHAPE = env.obs_shape
    return env


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=0)
    
    # Create a test env to get the observation shape
    print("Detecting observation shape...")
    test_env = DefensiveGFootball5v5(num_left_agents=5, num_right_agents=0)
    obs_shape = test_env.obs_shape
    print(f"Using observation shape: {obs_shape}")
    del test_env
    
    if args.self_play:
        register_env('gfootball_defensive', create_selfplay_env)
        
        policies = {
            'defensive_policy': (None, 
                spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
                spaces.Discrete(19), 
                {'gamma': 0.995}),
            'opponent_policy': (None,
                spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
                spaces.Discrete(19),
                {'gamma': 0.99}),
        }
        
        def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
            return 'defensive_policy' if agent_id.startswith('left_') else 'opponent_policy'
        
        print("=" * 60)
        print("SELF-PLAY MODE: Training defensive agents against themselves")
        print("=" * 60)
    else:
        register_env('gfootball_defensive', create_defensive_env)
        
        policies = {
            'defensive_policy': (None,
                spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
                spaces.Discrete(19),
                {'gamma': 0.995}),
        }
        
        def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
            return 'defensive_policy'
        
        print("=" * 60)
        print("DEFENSIVE TRAINING: 5v5 against built-in AI")
        print("=" * 60)
    
    config = {
        'env': 'gfootball_defensive',
        'lambda': 0.95,
        'kl_coeff': 0.2,
        'clip_rewards': False,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.005,
        'train_batch_size': 16000,
        'sgd_minibatch_size': 1024,
        'num_sgd_iter': 15,
        'num_workers': 4,
        'num_envs_per_worker': 4,
        'num_gpus': 0,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_share_layers': True,
        'lr': 1e-4,
        'log_level': 'WARN',
        'simple_optimizer': args.simple,
        'framework': 'torch',
        'disable_env_checking': True,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': policy_mapping_fn,
            'policies_to_train': ['defensive_policy'],
        },
    }
    
    print(f"\nTraining for {args.num_iters} iterations...")
    
    tune.run(
        'PPO',
        name='PPO_Defensive_5v5',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=args.checkpoint_freq,
        config=config,
    )