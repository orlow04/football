from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import gfootball.env as football_env
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from gymnasium import spaces
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
parser.add_argument('--episodes', type=int, default=5)
parser.add_argument('--record', action='store_true', help='Record videos')
parser.add_argument('--render', action='store_true', help='Render game')


def create_env_for_rllib(env_config):
    """Factory for RLlib environment registration."""
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    
    class RllibDefensive5v5(MultiAgentEnv):
        def __init__(self):
            super().__init__()
            self.num_agents = 5
            self.env = football_env.create_environment(
                env_name='5_vs_5',
                stacked=False,  # Must match training!
                representation='simple115v2',
                logdir='/tmp/rllib',
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=False,
                write_video=False,
                number_of_left_players_agent_controls=5,
                number_of_right_players_agent_controls=0)
            
            self._agent_ids = set([f'left_{i}' for i in range(5)])
            self._action_space = spaces.Discrete(19)
            self._observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
            
            self.observation_space = spaces.Dict({
                agent_id: self._observation_space for agent_id in self._agent_ids
            })
            self.action_space = spaces.Dict({
                agent_id: self._action_space for agent_id in self._agent_ids
            })

        def reset(self, *, seed=None, options=None):
            obs = self.env.reset()
            return {f'left_{i}': np.array(obs[i], dtype=np.float32) for i in range(5)}, \
                   {f'left_{i}': {} for i in range(5)}

        def step(self, action_dict):
            actions = [action_dict.get(f'left_{i}', 0) for i in range(5)]
            o, r, d, info = self.env.step(actions)
            
            obs = {f'left_{i}': np.array(o[i], dtype=np.float32) for i in range(5)}
            rewards = {f'left_{i}': float(r[i]) if isinstance(r, (list, np.ndarray)) else float(r) for i in range(5)}
            terminateds = {f'left_{i}': d for i in range(5)}
            terminateds['__all__'] = d
            truncateds = {f'left_{i}': False for i in range(5)}
            truncateds['__all__'] = False
            infos = {f'left_{i}': info for i in range(5)}
            
            return obs, rewards, terminateds, truncateds, infos
    
    return RllibDefensive5v5()


if __name__ == '__main__':
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    ray.init(num_gpus=0, ignore_reinit_error=True)
    register_env('gfootball_defensive', create_env_for_rllib)
    
    # Must match training observation shape (115 for stacked=False)
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
    act_space = spaces.Discrete(19)
    
    config = (
        PPOConfig()
        .environment('gfootball_defensive', disable_env_checking=True)
        .framework('torch')
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=0)
        .multi_agent(
            policies={
                'defensive_policy': (None, obs_space, act_space, {'gamma': 0.995}),
            },
            policy_mapping_fn=lambda agent_id, **kwargs: 'defensive_policy',
        )
    )
    
    algo = config.build()
    algo.restore(args.checkpoint)
    print("Model loaded!")
    
    # Create evaluation environment - must use stacked=False to match training
    os.makedirs('videos', exist_ok=True)
    
    eval_env = football_env.create_environment(
        env_name='5_vs_5',
        stacked=False,  # Must match training!
        representation='simple115v2',
        logdir='videos',
        write_goal_dumps=False,
        write_full_episode_dumps=args.record,
        render=args.render,
        write_video=args.record,
        number_of_left_players_agent_controls=5,
        number_of_right_players_agent_controls=0)
    
    print(f"\nAction Space: {eval_env.action_space}")
    print(f"Observation Space: {eval_env.observation_space}")
    print("-" * 40)
    
    print(f"\nRunning {args.episodes} episodes...")
    
    total_goals_scored = 0
    total_goals_conceded = 0
    
    for episode in range(args.episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get actions for all 5 agents
            actions = []
            for i in range(5):
                action = algo.compute_single_action(
                    np.array(obs[i], dtype=np.float32),
                    policy_id='defensive_policy',
                    explore=False
                )
                actions.append(action)
            
            obs, rew, done, info = eval_env.step(actions)
            total_reward += sum(rew) if isinstance(rew, (list, np.ndarray)) else rew
            steps += 1
        
        # Get final score
        if 'score' in info:
            left_score, right_score = info['score']
            total_goals_scored += left_score
            total_goals_conceded += right_score
            print(f"Episode {episode + 1}: Steps={steps}, Score={left_score}-{right_score}")
        else:
            print(f"Episode {episode + 1}: Steps={steps}, Reward={total_reward:.2f}")
    
    eval_env.close()
    algo.stop()
    ray.shutdown()
    
    print("\n" + "=" * 50)
    print("DEFENSIVE PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Goals Scored: {total_goals_scored}")
    print(f"Total Goals Conceded: {total_goals_conceded}")
    print(f"Goal Difference: {total_goals_scored - total_goals_conceded}")
    if args.record:
        print(f"\nVideos saved in 'videos' folder!")
    print("=" * 50)