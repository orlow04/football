import os
import argparse
import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import gfootball.env as football_env
import tempfile

# --- 1. DEFINIÇÃO DO AMBIENTE (Necessária para o Ray carregar o Checkpoint) ---
class DefensiveGFootball5v5(MultiAgentEnv):
    def __init__(self, num_left_agents=5, num_right_agents=0):
        super().__init__()
        self.num_left_agents = num_left_agents
        self.num_right_agents = num_right_agents
        self._num_agents = num_left_agents + num_right_agents
        
        # Criação básica do ambiente apenas para validar espaços
        self.env = football_env.create_environment(
            env_name='5_vs_5',
            stacked=False,  
            representation='simple115v2',
            logdir=os.path.join(tempfile.gettempdir(), 'defensive_export'),
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
        test_obs = self.env.reset()
        single_obs = test_obs[0] if self._num_agents > 1 else test_obs
        self.obs_shape = single_obs.shape
        
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        self.observation_space = spaces.Dict({id: self._observation_space for id in self._agent_ids})
        self.action_space = spaces.Dict({id: self._action_space for id in self._agent_ids})

    def reset(self, *, seed=None, options=None):
        return {}, {}
    def step(self, action_dict):
        return {}, {}, {}, {}, {}

def create_defensive_env(env_config):
    return DefensiveGFootball5v5(5, 0)

# --- 2. CONFIGURAÇÃO DE CAMINHOS ---

# O caminho exato que apareceu no seu log anterior:
CHECKPOINT_PATH = "/workspace/ray_results_resgatados/PPO_Defensive_5v5_FROM_ZERO/PPO_gfootball_defensive_c8b62_00000_0_2025-12-08_19-55-02/checkpoint_000012"

OUTPUT_DIR = "submission"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "best_model.pth")

def export_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERRO: Checkpoint não encontrado em: {CHECKPOINT_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Inicializando Ray...")
    ray.init(ignore_reinit_error=True)
    
    # --- A CORREÇÃO MÁGICA ESTÁ AQUI ---
    # Registramos o ambiente com O MESMO NOME usado no treino ('gfootball_defensive')
    # Isso engana o Ray para ele achar que está tudo bem e carregar os pesos.
    register_env('gfootball_defensive', create_defensive_env)
    print("Ambiente registrado com sucesso.")

    print(f"Carregando checkpoint...")
    algo = PPO.from_checkpoint(CHECKPOINT_PATH)
    
    print("Extraindo política...")
    policy = algo.get_policy("defensive_policy")
    torch_model = policy.model
    
    print("Salvando modelo PyTorch...")
    # Salvamos apenas o state_dict para ser leve
    torch.save(torch_model.state_dict(), OUTPUT_FILE)
    
    print(f"\n✅ SUCESSO! Modelo salvo em: {OUTPUT_FILE}")
    print("Próximos passos:")
    print("1. Verifique se a pasta 'submission' tem os arquivos 'agent.py' e 'best_model.pth'.")
    print("2. Zipe a pasta ou envie conforme solicitado.")

    ray.shutdown()

if __name__ == "__main__":
    export_model()