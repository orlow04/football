# Arquivo: /Users/orlow/dev/rl/final/getafe-ball/export_weights.py
import ray
from ray.rllib.algorithms.ppo import PPO
import torch
import os

# --- CONFIGURE AQUI O CAMINHO DO SEU CHECKPOINT ---
# Copie o caminho relativo da sua pasta ray_results
# Exemplo baseado na sua árvore:
CHECKPOINT_PATH = os.path.join(
    os.getcwd(),
    "ray_results/PPO_Defensive_5v5/PPO_gfootball_defensive_ea7a3_00000_0_2025-12-08_05-58-27/checkpoint_000200"
)

OUTPUT_DIR = "submission"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "best_model.pth")

def export_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERRO: Checkpoint não encontrado em: {CHECKPOINT_PATH}")
        return

    # Cria a pasta submission se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Inicializando Ray...")
    ray.init(ignore_reinit_error=True)
    
    print(f"Carregando checkpoint: {CHECKPOINT_PATH}")
    algo = PPO.from_checkpoint(CHECKPOINT_PATH)
    policy = algo.get_policy("defensive_policy")
    torch_model = policy.model
    
    print("Extraindo pesos...")
    torch.save(torch_model.state_dict(), OUTPUT_FILE)
    print(f" SUCESSO! Modelo salvo em: {OUTPUT_FILE}")
    print("Agora você pode zipar a pasta 'submission' ou entregar os arquivos dela.")

if __name__ == "__main__":
    export_model()