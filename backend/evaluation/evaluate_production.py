import os
import sys
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import training components
from src.training.traffic_env import TrafficEnv
from src.training.train_ppo import TrueTemporalLSTMExtractor, TrafficRegime
from src.utils.arrival_rate_converter import get_hourly_rates

class ProductionEvaluator:
    def __init__(self, model_dir: str, data_path: str, verbose: bool = True):
        self.model_dir = model_dir
        self.data_path = data_path
        self.verbose = verbose
        
        self.policies = {}
        self.models_loaded = False
        
        # Load policies
        self._load_policies()
        
    def _load_policies(self):
        """Load all 3 regime policies."""
        regimes = {
            TrafficRegime.NIGHT: "policy_NIGHT.zip",
            TrafficRegime.OFF_PEAK: "policy_OFF_PEAK.zip",
            TrafficRegime.PEAK: "policy_PEAK.zip"
        }
        
        if self.verbose:
            print(f"📦 Loading policies from {self.model_dir}...")
            
        for regime, filename in regimes.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            try:
                # We don't need to pass env here strictly for loading, but it's good practice
                # Use custom objects map just in case, though import should handle it
                self.policies[regime] = PPO.load(
                    path, 
                    custom_objects={
                        "learning_rate": 0.0,
                        "lr_schedule": lambda _: 0.0,
                        "clip_range": lambda _: 0.0
                    }
                )
                if self.verbose:
                    print(f"   ✓ Loaded {TrafficRegime.get_name(regime)} policy")
            except Exception as e:
                raise RuntimeError(f"Failed to load {filename}: {e}")
                
        self.models_loaded = True

    def _create_eval_env(self, arrival_rates):
        """Create an environment wrapped exactly like training."""
        def make_env():
            env = TrafficEnv(
                max_steps=3600,
                saturation_flow_rate=0.53,
                lanes_per_direction=2,
                track_waiting_time=True,
            )
            env.reset(arrival_rates=arrival_rates) # Initialize with rates
            return Monitor(env)

        # Wrap in DummyVecEnv
        env = DummyVecEnv([make_env])
        
        # Apply VecFrameStack (History Len = 10, same as training)
        env = VecFrameStack(env, n_stack=10)
        
        return env

    def evaluate_hour(self, hour: int, n_episodes: int = 5) -> dict:
        """Evaluate a specific hour using the appropriate policy."""
        # Determine regime and policy
        regime = TrafficRegime.from_hour(hour)
        policy = self.policies[regime]
        regime_name = TrafficRegime.get_name(regime)
        
        if self.verbose:
            print(f"   Hour {hour:02d}:00 - Regime: {regime_name} - Model: policy_{regime_name}")
            
        # Get arrival rates
        rates = get_hourly_rates(self.data_path, hour)
        
        # Create environment
        env = self._create_eval_env(rates)
        
        episode_metrics = []
        
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            
            # Since VecEnv auto-resets, we need to manually track done
            # But here we just run for max_steps which is handled by TrafficEnv
            # TrafficEnv doesn't return done=True until max_steps
            
            # We need to extract the underlying env to get accurate metrics
            # VecFrameStack -> DummyVecEnv -> Monitor -> TrafficEnv
            underlying_env = env.envs[0].env 
            
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                total_reward += rewards[0]
                
                if dones[0]:
                    break
            
            metrics = underlying_env.get_metrics()
            metrics['episode_reward'] = total_reward
            metrics['hour'] = hour
            metrics['regime'] = regime_name
            episode_metrics.append(metrics)
            
        env.close()
        
        # Aggregate
        aggregated = {}
        keys = ['avg_queue_length', 'avg_vehicle_delay', 'total_throughput', 'episode_reward', 'queue_imbalance']
        
        for k in keys:
            vals = [m[k] for m in episode_metrics]
            aggregated[f"{k}_mean"] = np.mean(vals)
            aggregated[f"{k}_std"] = np.std(vals)
            
        aggregated['hour'] = hour
        aggregated['regime'] = regime_name
        aggregated['n_episodes'] = n_episodes
        
        return aggregated

    def evaluate_full_day(self, n_episodes: int = 5, output_dir: str = "results"):
        """Run full day evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🚦 STARTING FULL DAY EVALUATION (Episodes/Hour: {n_episodes})")
        print("=" * 60)
        
        results = []
        for hour in range(24):
            res = self.evaluate_hour(hour, n_episodes)
            results.append(res)
            
            if self.verbose:
                print(f"      Queue: {res['avg_queue_length_mean']:.2f} | Delay: {res['avg_vehicle_delay_mean']:.2f}s | Reward: {res['episode_reward_mean']:.2f}")

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/rl_evaluation.csv", index=False)
        
        self._generate_summary(df, output_dir)
        print("=" * 60)
        print(f"✅ Evaluation Complete. Results saved to {output_dir}/")

    def _generate_summary(self, df: pd.DataFrame, output_dir: str):
        summary_path = f"{output_dir}/rl_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("RL PRODUCTION MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL:\n")
            f.write(f"Average Queue: {df['avg_queue_length_mean'].mean():.2f}\n")
            f.write(f"Average Delay: {df['avg_vehicle_delay_mean'].mean():.2f}s\n")
            f.write(f"Total Throughput (Avg): {df['total_throughput_mean'].mean():.0f}\n\n")
            
            f.write("BY REGIME:\n")
            for regime in df['regime'].unique():
                regime_df = df[df['regime'] == regime]
                f.write(f"  {regime}:\n")
                f.write(f"    Queue: {regime_df['avg_queue_length_mean'].mean():.2f}\n")
                f.write(f"    Delay: {regime_df['avg_vehicle_delay_mean'].mean():.2f}s\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory containing policy_*.zip files")
    parser.add_argument("--data", required=True, help="Arrival rates CSV")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per hour")
    
    args = parser.parse_args()
    
    evaluator = ProductionEvaluator(args.model_dir, args.data)
    evaluator.evaluate_full_day(args.episodes, args.output)

if __name__ == "__main__":
    main()
