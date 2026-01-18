"""
SUMO-RL Environment
Custom environment for traffic signal control
"""
from sumo_rl import SumoEnvironment
import gymnasium as gym
from app.config import settings
import os


class TrafficEnv:
    """Wrapper for SUMO-RL environment"""
    
    def __init__(self, use_gui: bool = False):
        self.use_gui = use_gui
        self.env = None
        
    def create_env(self):
        """Create SUMO-RL environment"""
        try:
            self.env = SumoEnvironment(
                net_file=settings.NETWORK_FILE,
                route_file=settings.ROUTE_FILE,
                use_gui=self.use_gui,
                num_seconds=settings.SIMULATION_TIME,
                min_green=settings.MIN_GREEN,
                max_green=settings.MAX_GREEN,
                yellow_time=settings.YELLOW_TIME,
                reward_fn=self._custom_reward,
                sumo_seed=42,
                fixed_ts=False,
                sumo_warnings=False
            )
            
            return self.env
            
        except Exception as e:
            print(f"Error creating environment: {e}")
            return None
    
    def _custom_reward(self, traffic_signal):
        """
        Custom reward function
        Minimize waiting time and queue length
        
        Args:
            traffic_signal: Traffic signal object from SUMO-RL
            
        Returns:
            float: Reward value (negative for minimization)
        """
        # Get metrics
        queue_length = sum(traffic_signal.get_lanes_queue())
        waiting_time = sum(traffic_signal.get_lanes_waiting_time())
        
        # Combined reward (negative because we want to minimize)
        reward = -(waiting_time + queue_length * 10)
        
        return reward
    
    def reset(self):
        """Reset environment"""
        if self.env:
            return self.env.reset()
        return None
    
    def step(self, action):
        """Take action in environment"""
        if self.env:
            return self.env.step(action)
        return None, None, None, None
    
    def close(self):
        """Close environment"""
        if self.env:
            self.env.close()


def create_training_env():
    """Create environment for training (returns SumoEnvironment directly)"""
    from app.config import settings
    import os
    
    # Verify files exist
    if not os.path.exists(settings.NETWORK_FILE):
        raise FileNotFoundError(f"Network file not found: {settings.NETWORK_FILE}")
    if not os.path.exists(settings.ROUTE_FILE):
        raise FileNotFoundError(f"Route file not found: {settings.ROUTE_FILE}")
    
    # Create SUMO-RL environment directly
    env = SumoEnvironment(
        net_file=settings.NETWORK_FILE,
        route_file=settings.ROUTE_FILE,
        use_gui=False,
        num_seconds=settings.SIMULATION_TIME,
        delta_time=5,  # Agent decides every 5 seconds
        yellow_time=settings.YELLOW_TIME,
        min_green=settings.MIN_GREEN,
        max_green=settings.MAX_GREEN,
        sumo_seed=42,
        fixed_ts=False,  # RL controls the signals
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log --no-warnings"
    )
    
    return env
