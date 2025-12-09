import torch
import torch.nn as nn
import numpy as np
import os

class Player(object):
    def __init__(self, player_config, env_config):
        self._player_config = player_config
        self._env_config = env_config

    def take_action(self, observation):
        raise NotImplementedError

class FootballAgent(Player):
    def __init__(self, player_config, env_config):
        # Inicializa a classe pai que definimos acima
        Player.__init__(self, player_config, env_config)
        
        # Caminho do modelo (mesma pasta do script)
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        
        # Detecta dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define a arquitetura (tem que bater com o treino PPO)
        self.model = nn.Sequential(
            nn.Linear(115, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 19)
        )
        
        try:
            # Carrega os pesos
            print(f"Carregando agente de: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Carrega no modelo
            self.model.load_state_dict(state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            print("✅ Pesos carregados com sucesso!")
        except Exception as e:
            print(f"❌ ERRO ao carregar pesos: {e}")
            self.model = None

    def take_action(self, observation):
        if self.model is None:
            return 0 # Ação nula se o modelo falhar
            
        # Garante que a observação seja um array numpy float32
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
            
        # Converte para tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            action = torch.argmax(logits).item()
            
        return action

# Função fábrica obrigatória para o torneio
def agent_factory(player_config, env_config):
    return FootballAgent(player_config, env_config)