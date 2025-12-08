from __future__ import absolute_import, division, print_function

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
from ray.rllib.algorithms.callbacks import DefaultCallbacks

parser = argparse.ArgumentParser()
parser.add_argument('--num-iters', type=int, default=5000)
parser.add_argument('--checkpoint-freq', type=int, default=50)
parser.add_argument('--self-play', action='store_true', help='Enable self-play training')
parser.add_argument('--simple', action='store_true')

# --- CALLBACK PARA O SELF-PLAY (Oponente Evolutivo) ---
class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # A cada 20 iterações, o oponente copia a inteligência do agente atual
        if result["training_iteration"] > 0 and result["training_iteration"] % 20 == 0:
            # Pega os pesos da política defensiva (principal)
            weights = algorithm.get_policy("defensive_policy").get_weights()
            # Transfere para a política do oponente
            algorithm.get_policy("opponent_policy").set_weights(weights)
            
            win_rate = result['env_runners']['policy_reward_mean'].get('defensive_policy', 0)
            print(f"\n[SELF-PLAY] Oponente atualizado na iteração {result['training_iteration']}! (Score Médio Atual: {win_rate:.2f})")

class DefensiveGFootball5v5(MultiAgentEnv):
    def __init__(self, num_left_agents=5, num_right_agents=0):
        super().__init__()
        self.num_left_agents = num_left_agents
        self.num_right_agents = num_right_agents
        self._num_agents = num_left_agents + num_right_agents
        
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
        
        self._agent_ids = set()
        for i in range(num_left_agents): self._agent_ids.add(f'left_{i}')
        for i in range(num_right_agents): self._agent_ids.add(f'right_{i}')
        
        self._action_space = spaces.Discrete(19)
        
        # Detect obs shape
        test_obs = self.env.reset()
        single_obs = test_obs[0] if self._num_agents > 1 else test_obs
        self.obs_shape = single_obs.shape
        
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        self.observation_space = spaces.Dict({id: self._observation_space for id in self._agent_ids})
        self.action_space = spaces.Dict({id: self._action_space for id in self._agent_ids})
        
        self.prev_ball_owned_team = None
        self.goals_conceded = 0
        self.goals_scored = 0
        self.prev_score = (0, 0)

    def _get_defensive_rewards(self, obs_raw, base_reward, done, info):
        rewards = {}
        try:
            if isinstance(obs_raw, (list, np.ndarray)):
                obs = obs_raw[0] if len(obs_raw) > 0 else obs_raw
                ball_x = obs[0] # Posição X da bola
                ball_owned_team = info.get('ball_owned_team', -1)
                
                left_score, right_score = info.get('score', (0,0))
                score_diff_step = (left_score - self.prev_score[0]) - (right_score - self.prev_score[1])
                self.prev_score = (left_score, right_score)
        except:
            ball_x = 0; ball_owned_team = -1; score_diff_step = 0
        
        for agent_id in self._agent_ids:
            reward = 0.0
            is_left_team = agent_id.startswith('left_')
            
            # 1. PENALIDADE MAXIMA: Tomar Gol
            if (is_left_team and score_diff_step < 0) or (not is_left_team and score_diff_step > 0):
                reward -= 10.0 # TRAUMA
                if is_left_team: self.goals_conceded += 1
            
            # 2. RECOMPENSA PEQUENA: Fazer Gol (necessário para vencer, mas secundário)
            if (is_left_team and score_diff_step > 0) or (not is_left_team and score_diff_step < 0):
                reward += 2.0 

            # 3. INTERCEPTAÇÃO (Roubada de bola)
            if self.prev_ball_owned_team is not None:
                if is_left_team and self.prev_ball_owned_team == 1 and ball_owned_team == 0:
                    reward += 0.5 
                elif not is_left_team and self.prev_ball_owned_team == 0 and ball_owned_team == 1:
                    reward += 0.5

            # 4. CLEARANCE (Afastar o perigo) - MUDANÇA IMPORTANTE
            # Se a bola estava na defesa (< -0.5) e agora está mais a frente, ganha ponto.
            # Evita que o agente fique tocando a bola na defesa.
            if is_left_team and ball_owned_team == 0:
                if ball_x > -0.1: # Passou do meio campo ou quase
                    reward += 0.005 # Incentivo constante para manter a bola longe

            # 5. CLEAN SHEET (Sobrevivência)
            # Ganha um pouquinho a cada step que NÃO toma gol
            reward += 0.001 
            
            rewards[agent_id] = reward
        
        self.prev_ball_owned_team = ball_owned_team
        return rewards

    def reset(self, *, seed=None, options=None):
        original_obs = self.env.reset()
        self.prev_ball_owned_team = None
        self.prev_score = (0, 0)
        self.goals_conceded = 0
        obs = {}
        idx = 0
        for i in range(self.num_left_agents):
            obs[f'left_{i}'] = np.array(original_obs[idx], dtype=np.float32)
            idx += 1
        for i in range(self.num_right_agents):
            obs[f'right_{i}'] = np.array(original_obs[idx], dtype=np.float32)
            idx += 1
        return obs, {k: {} for k in obs.keys()}

    def step(self, action_dict):
        actions = [action_dict.get(f'left_{i}', 0) for i in range(self.num_left_agents)] + \
                  [action_dict.get(f'right_{i}', 0) for i in range(self.num_right_agents)]
        
        o, r, d, info = self.env.step(actions)
        def_rewards = self._get_defensive_rewards(o, r, d, info)
        
        obs = {}
        idx = 0
        for i in range(self.num_left_agents):
            obs[f'left_{i}'] = np.array(o[idx], dtype=np.float32); idx += 1
        for i in range(self.num_right_agents):
            obs[f'right_{i}'] = np.array(o[idx], dtype=np.float32); idx += 1
            
        terminateds = {k: d for k in obs.keys()}; terminateds['__all__'] = d
        truncateds = {k: False for k in obs.keys()}; truncateds['__all__'] = False
        return obs, def_rewards, terminateds, truncateds, {k: info for k in obs.keys()}

class SelfPlayGFootball5v5(DefensiveGFootball5v5):
    def __init__(self): super().__init__(num_left_agents=5, num_right_agents=5)

# --- CONFIGURAÇÃO DO RAY ---
def create_selfplay_env(env_config): return SelfPlayGFootball5v5()
def create_defensive_env(env_config): return DefensiveGFootball5v5(5, 0)

if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=1) # Ative a GPU se tiver

    test_env = DefensiveGFootball5v5(5, 0)
    obs_shape = test_env.obs_shape
    
    # Configuração de Políticas
    policies = {
        'defensive_policy': (None, spaces.Box(-np.inf, np.inf, obs_shape, np.float32), spaces.Discrete(19), {}),
    }
    
    if args.self_play:
        register_env('gfootball_defensive', create_selfplay_env)
        # Adiciona a política do oponente para o Self-Play
        policies['opponent_policy'] = (None, spaces.Box(-np.inf, np.inf, obs_shape, np.float32), spaces.Discrete(19), {})
        
        def policy_mapping_fn(agent_id, **kwargs):
            return 'defensive_policy' if agent_id.startswith('left_') else 'opponent_policy'
    else:
        register_env('gfootball_defensive', create_defensive_env)
        def policy_mapping_fn(agent_id, **kwargs): return 'defensive_policy'

    config = {
        'env': 'gfootball_defensive',
        'framework': 'torch',
        'num_workers': 2,
        'num_envs_per_worker': 2,
        'num_gpus': 1, # Use 1 se tiver GPU
        'train_batch_size': 4000,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': policy_mapping_fn,
            'policies_to_train': ['defensive_policy'], # Só treina o nosso, o oponente copia
        },
        # IMPORTANTE: Registra o callback do Self-Play
        'callbacks': SelfPlayUpdateCallback if args.self_play else None
    }

    CHECKPOINT_PATH = "/Users/orlow/dev/rl/final/getafe-ball/ray_results/PPO_Defensive_5v5/PPO_gfootball_defensive_ea7a3_00000_0_2025-12-08_05-58-27/checkpoint_000016"
    
    tune.run(
        'PPO',
        name='PPO_Defensive_5v5',
        restore=CHECKPOINT_PATH,
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=args.checkpoint_freq,
        config=config,
    )