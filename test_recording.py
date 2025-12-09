import sys
import os

# --- BLOCO DE CORREÇÃO DE IMPORTAÇÃO ---
# Pega o caminho absoluto da pasta onde ESTE script (visualize_agent.py) está
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona ao sys.path
sys.path.append(current_dir)

print(f"Diretório de trabalho: {os.getcwd()}")
print(f"Caminho do script: {current_dir}")
print(f"Arquivos na pasta submission: {os.listdir(os.path.join(current_dir, 'submission'))}")
# ---------------------------------------

import gfootball.env as football_env
import numpy as np

try:
    from submission.agent import agent_factory
    print("✅ SUCESSO! Agente importado.")
except ImportError as e:
    print(f"\n❌ ERRO FATAL DE IMPORTAÇÃO: {e}")
    print("Verifique se existe um arquivo '__init__.py' (vazio) dentro da pasta submission.")
    exit(1)

# Configurações de vídeo
video_path = os.path.join(os.getcwd(), 'videos_agent')

env = football_env.create_environment(
    env_name="5_vs_5", 
    

    stacked=False, 
    
    representation='simple115v2',
    logdir=video_path, 
    write_goal_dumps=False, 
    write_full_episode_dumps=True,  # Grava o jogo inteiro
    render=False,                   # No Docker, render=True costuma falhar. O vídeo é salvo no disco.
    write_video=True,
    
    # Garante que controlamos os 5 jogadores (como no treino)
    number_of_left_players_agent_controls=5
)

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
print("--------------------------------")

# Inicializa seu Agente
# O environment config pode ser None para testes simples
agent = agent_factory(None, None)

print("🎥 Iniciando partida de exibição...")
obs = env.reset()
steps = 0
total_reward = 0

while True:
    actions = []
    
    # CORREÇÃO: Tratamento robusto para matrizes NumPy (5, 115)
    # Se for um array e tiver dimensão 5 no início, é multi-agent
    if hasattr(obs, 'shape') and obs.shape[0] == 5:
        # Itera sobre as 5 linhas da matriz (uma obs por jogador)
        for i in range(5):
            player_obs = obs[i] # Pega a visão do jogador i
            action = agent.take_action(player_obs)
            actions.append(action)
            
    # Fallback: Se for lista (formato antigo de alguns envs)
    elif isinstance(obs, list) and len(obs) == 5:
        for player_obs in obs:
            action = agent.take_action(player_obs)
            actions.append(action)
            
    # Fallback 2: Single agent
    else:
        action = agent.take_action(obs)
        actions = [action]

    # Envia as 5 ações para o ambiente
    obs, rew, done, info = env.step(actions)
    
    # Soma a recompensa (pega a do primeiro jogador ou soma se for lista)
    if isinstance(rew, (list, np.ndarray)):
        step_reward = rew[0] 
    else:
        step_reward = rew
        
    total_reward += step_reward
    
    steps += 1
    if steps % 100 == 0:
        print(f"Step {steps} | Score: {info.get('score', '0-0')} | Reward Acumulado: {total_reward:.2f}")
    
    if done:
        print(f"Fim de jogo! Resultado Final: {info.get('score', 'Unknown')}")
        break

env.close()
print(f"✅ Jogo finalizado em {steps} passos.")
print(f"📂 Verifique os vídeos gerados na pasta: {video_path}")


