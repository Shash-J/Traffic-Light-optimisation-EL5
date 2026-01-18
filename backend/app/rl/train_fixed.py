"""
IMPROVED Production Training Script - V3
=========================================
Key improvements over V2:
1. Normalized reward scaling for stable learning
2. Adaptive phase switch penalty
3. Removed terminal reward (interferes with TD learning)
4. Added reward statistics tracking
5. Better hyperparameter choices for this problem
"""
import os
import sys
import argparse
from datetime import datetime
from collections import deque

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(script_dir)
backend_dir = os.path.dirname(app_dir)
sys.path.insert(0, backend_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import gymnasium as gym
from sumo_rl import SumoEnvironment
import traci


class EpisodeTrackingCallback(BaseCallback):
    """Enhanced callback with reward statistics and convergence detection."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_avg_queues = []
        self.episode_avg_waits = []
        self.best_mean_reward = -np.inf
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = datetime.now()
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Extract custom metrics if available
                if "avg_queue" in info:
                    self.episode_avg_queues.append(info["avg_queue"])
                if "avg_wait" in info:
                    self.episode_avg_waits.append(info["avg_wait"])
                
                ep_num = len(self.episode_rewards)
                
                print(f"\n📊 Episode {ep_num}:")
                print(f"   Reward: {ep_reward:.2f}")
                print(f"   Length: {ep_length} steps")
                
                if "avg_queue" in info:
                    print(f"   Avg Queue: {info['avg_queue']:.1f} vehicles")
                if "avg_wait" in info:
                    print(f"   Avg Wait: {info['avg_wait']:.1f} seconds")
                
                # Track best performance
                if len(self.episode_rewards) >= 10:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"   🏆 New best 10-ep mean: {mean_reward:.2f}")
                
                # Convergence check every 100 episodes
                if ep_num % 100 == 0 and ep_num >= 200:
                    recent_100 = np.mean(self.episode_rewards[-100:])
                    prev_100 = np.mean(self.episode_rewards[-200:-100])
                    improvement = recent_100 - prev_100
                    
                    print(f"\n{'='*60}")
                    print(f"📈 CONVERGENCE CHECK @ Episode {ep_num}:")
                    print(f"   Episodes {ep_num-200}-{ep_num-100}: {prev_100:.2f}")
                    print(f"   Episodes {ep_num-100}-{ep_num}: {recent_100:.2f}")
                    print(f"   Improvement: {improvement:+.2f}")
                    
                    if abs(improvement) < 10:
                        print(f"   ⚠️ Learning may be plateauing")
                    elif improvement > 20:
                        print(f"   ✅ Strong improvement detected!")
                    
                    print(f"{'='*60}\n")
                
        return True


class ImprovedSumoWrapper(gym.Wrapper):
    """
    Improved wrapper with normalized rewards and better metrics.
    
    Key improvements:
    1. Reward normalization using running statistics
    2. Adaptive phase switch penalty
    3. Separate tracking of queue and wait metrics
    4. No terminal reward (better for TD learning)
    """
    
    def __init__(self, env, max_steps=120, debug=False):
        super().__init__(env)
        self.agent_id = None
        self.current_step = 0
        self.max_steps = max_steps
        self.debug = debug
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_count = 0
        self.prev_action = None
        
        # Metrics tracking
        self.step_queues = []
        self.step_waits = []
        
        # Reward normalization
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0
        self.reward_std = 1
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.episode_reward = 0
        self.prev_action = None
        self.step_queues = []
        self.step_waits = []
        
        res = self.env.reset()
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}
        
        if isinstance(obs, dict):
            self.agent_id = list(obs.keys())[0]
            obs = obs[self.agent_id]
        
        if self.debug:
            print(f"\n🔄 Episode {self.episode_count + 1} started")
        
        return np.array(obs, dtype=np.float32), info
    
    def _calculate_reward(self, action):
        """
        Calculate normalized pain-based reward.
        """
        try:
            total_waiting_time = 0.0
            total_queue_length = 0
            
            # Get metrics from SUMO
            for vehicle_id in traci.vehicle.getIDList():
                total_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)
            
            tl_ids = traci.trafficlight.getIDList()
            if tl_ids:
                tl_id = tl_ids[0]
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane_id in set(lanes):
                    total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
            
            # Track for episode statistics
            self.step_queues.append(total_queue_length)
            self.step_waits.append(total_waiting_time)
            
            # Base reward (negative = bad)
            # Normalize by expected values to keep scale reasonable
            # Typical peak: 20-50 queue, 100-300 wait time
            normalized_queue = total_queue_length / 30.0
            normalized_wait = total_waiting_time / 150.0
            
            reward = -(normalized_queue + normalized_wait)
            
            # Adaptive phase switch penalty
            # Scale with current congestion level
            if self.prev_action is not None and action != self.prev_action:
                congestion_factor = min(normalized_queue, 3.0)
                penalty = 0.5 * (1 + congestion_factor)  # 0.5 to 2.0
                reward -= penalty
            
            # Update normalization statistics
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = np.std(self.reward_history) + 1e-8
            
            # Normalize reward for stable learning
            normalized_reward = (reward - self.reward_mean) / self.reward_std
            
            # Clip to reasonable range
            normalized_reward = float(np.clip(normalized_reward, -10, 10))
            
            if self.debug and self.current_step % 20 == 0:
                print(f"   🚦 queue={total_queue_length}, wait={total_waiting_time:.1f}s")
                print(f"      raw_reward={reward:.2f}, normalized={normalized_reward:.2f}")
            
            return normalized_reward, total_waiting_time, total_queue_length
            
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Reward calc error: {e}")
            return -5.0, 0, 0  # Moderate penalty on error
    
    def step(self, action):
        self.current_step += 1
        
        # Execute action
        if self.agent_id:
            res = self.env.step({self.agent_id: action})
        else:
            res = self.env.step(action)
        
        # Parse result
        if len(res) == 5:
            obs, _, terminated, truncated, info = res
        else:
            obs, _, done, info = res
            terminated = done
            truncated = False
        
        # Extract from dict if needed
        if isinstance(obs, dict):
            obs = obs[self.agent_id]
        if isinstance(terminated, dict):
            terminated = terminated.get(self.agent_id, False)
            if "__all__" in res[2]:
                terminated = res[2]["__all__"]
        if isinstance(truncated, dict):
            truncated = truncated.get(self.agent_id, False)
        
        # Force truncation at max steps
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Calculate normalized reward
        reward, wait_time, queue_len = self._calculate_reward(action)
        
        # Sanitize
        if np.isnan(reward) or np.isinf(reward):
            reward = -5.0
        
        self.episode_reward += reward
        self.prev_action = action
        
        # Add episode statistics to info
        if terminated or truncated:
            self.episode_count += 1
            info["avg_queue"] = np.mean(self.step_queues) if self.step_queues else 0
            info["avg_wait"] = np.mean(self.step_waits) if self.step_waits else 0
            
            if self.debug:
                print(f"   ✅ Episode {self.episode_count}:")
                print(f"      Total reward: {self.episode_reward:.2f}")
                print(f"      Avg queue: {info['avg_queue']:.1f}")
                print(f"      Avg wait: {info['avg_wait']:.1f}s")
        
        obs = np.array(obs, dtype=np.float32)
        
        return obs, reward, terminated, truncated, info


def create_env(net_file: str, route_file: str, gui: bool = False, debug: bool = False):
    """Create improved SUMO environment."""
    
    sumo_env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=gui,
        num_seconds=600,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        sumo_seed=42,
        fixed_ts=False,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )
    
    wrapped_env = ImprovedSumoWrapper(sumo_env, max_steps=120, debug=debug)
    monitored_env = Monitor(wrapped_env)
    
    return monitored_env


def train_model(timesteps: int = 200000, debug: bool = False):
    """Train with improved environment and hyperparameters."""
    
    print("=" * 70)
    print("🚦 IMPROVED RL TRAINING - Bangalore Traffic Signal Control")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timesteps: {timesteps:,}")
    print()
    
    # Paths
    net_file = os.path.join(app_dir, "sumo", "network", "network.net.xml")
    route_file = os.path.join(app_dir, "sumo", "network", "routes_peak.rou.xml")
    save_dir = os.path.join(backend_dir, "models", "checkpoints")
    tensorboard_dir = os.path.join(backend_dir, "tensorboard_logs", "ppo_improved")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Verify files
    for f, name in [(net_file, "Network"), (route_file, "Routes")]:
        if not os.path.exists(f):
            print(f"❌ {name} file not found: {f}")
            return None
        print(f"✓ {name}: {os.path.basename(f)}")
    
    print()
    
    # Create environment
    print("📦 Creating improved SUMO environment...")
    env = create_env(net_file, route_file, gui=False, debug=debug)
    
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()
    
    # PPO with tuned hyperparameters for traffic control
    print("🧠 Initializing PPO with improved hyperparameters...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,        # Slightly higher for faster initial learning
        n_steps=2048,
        batch_size=128,            # Larger batches for stability
        n_epochs=10,
        gamma=0.95,                # Lower gamma - traffic state changes quickly
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,             # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])  # Larger networks
        ),
        verbose=1,
        tensorboard_log=tensorboard_dir,
    )
    print("   ✓ Model created")
    print()
    
    # Callbacks
    episode_callback = EpisodeTrackingCallback(verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="ppo_improved"
    )
    
    # Train
    print("🏋️ Starting training...")
    print("-" * 70)
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[episode_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    finally:
        # Save final model
        final_path = os.path.join(save_dir, f"ppo_improved_final_{timesteps}.zip")
        model.save(final_path)
        print(f"\n✅ Model saved: {final_path}")
        
        env.close()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved RL Training")
    parser.add_argument("--timesteps", type=int, default=200000, 
                        help="Training timesteps (default: 200k)")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    args = parser.parse_args()
    
    train_model(timesteps=args.timesteps, debug=args.debug)