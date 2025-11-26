import os, json
import pandas as pd
import wandb
from wandb.sdk.lib import runid

# Change to your actual trial path if different
TRIAL_DIR = "/Users/orlow/dev/rl/final/getafe-ball/ray_results/PPO_gfootball_defensive_a8710_00000_0_2025-11-26_19-46-53"

def main():
    run = wandb.init(project="grf-defensive-5v5", group="ppo-rllib", job_type="import", config={"trial_dir": TRIAL_DIR})

    # Upload progress.csv as a W&B Table and also log key columns as scalars per row
    progress_path = os.path.join(TRIAL_DIR, "progress.csv")
    if os.path.exists(progress_path):
        df = pd.read_csv(progress_path)
        wandb.log({"progress_table": wandb.Table(dataframe=df)})
        # Stream common metrics
        for _, row in df.iterrows():
            payload = {}
            for col in ["training_iteration", "episode_reward_mean", "episode_len_mean", "timesteps_total", "time_total_s", "policy_loss", "vf_loss", "entropy", "rollout_fragment_length"]:
                if col in df.columns:
                    payload[col] = row[col]
            if payload:
                wandb.log(payload)
        print("Uploaded progress.csv")
    else:
        print("progress.csv not found:", progress_path)

    # Upload result.json (Ray Tune JSON lines)
    result_path = os.path.join(TRIAL_DIR, "result.json")
    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        with open(result_path) as f:
            for line in f:
                try:
                    wandb.log(json.loads(line))
                except Exception:
                    pass
        print("Uploaded result.json")
    else:
        print("result.json not found or empty:", result_path)

    # Attach checkpoint metadata
    ckpt_dir = os.path.join(TRIAL_DIR, "checkpoint_000100")
    if os.path.isdir(ckpt_dir):
        artifact = wandb.Artifact("checkpoint_000100", type="model")
        artifact.add_dir(ckpt_dir)
        wandb.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    main()