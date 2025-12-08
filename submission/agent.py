# Arquivo: /Users/orlow/dev/rl/final/getafe-ball/submission/agent.py
import torch
import torch.nn as nn
import numpy as np
import os
from gfootball.env.players.player import Player

class FootballAgent(Player):
    def __init__(self, player_config, env_config):
        Player.__init__(self, player_config, env_config)
        
        # Procura o best_model.pth na MESMA pasta deste script
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        
        # Detecta GPU/CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Recria a estrutura (Compatível com o Default do RLlib PPO)
        # Se você mudou a arquitetura no config do treino, altere aqui também.
        # Padrão RLlib: 2 camadas ocultas de 256
        self.model = nn.Sequential(
            nn.Linear(115, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 19)
        )
        
        try:
            print(f"Carregando modelo de: {model_path}")
            # Carrega os pesos
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Limpeza de chaves do Ray (se necessário)
            clean_state_dict = {}
            for k, v in state_dict.items():
                if "value_branch" in k: continue
                # Remove prefixos estranhos que o Ray adiciona às vezes
                new_key = k.replace("_hidden_layers.0.", "0.") \
                           .replace("_hidden_layers.1.", "2.") \
                           .replace("_logits.", "4.")
                clean_state_dict[new_key] = v
            
            # Tenta carregar (strict=False permite ignorar diferenças pequenas)
            self.model.load_state_dict(clean_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("Agente PRONTO para jogar!")
            
        except Exception as e:
            print(f"ERRO CRÍTICO ao carregar agente: {e}")
            self.model = None

    def take_action(self, observation):
        if self.model is None:
            return 0 # Ação aleatória/Idle se falhar
            
        # Converte observação para tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            action = torch.argmax(logits).item()
            
        return action

def agent_factory(player_config, env_config):
    return FootballAgent(player_config, env_config)