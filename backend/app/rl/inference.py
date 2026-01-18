"""
RL Model Inference
Load trained model and use for live simulation control
Supports time-based policy selection (PEAK, OFF_PEAK, NIGHT)
"""
from stable_baselines3 import PPO, DQN
import os
from datetime import datetime
from app.config import settings
from app.sumo.traci_handler import traci_handler
import traci
import numpy as np


class RLAgent:
    """RL Agent for traffic signal control with time-based policy selection"""
    
    # Time ranges for different policies (24-hour format)
    PEAK_HOURS = [(7, 10), (17, 20)]      # 7-10 AM and 5-8 PM
    OFF_PEAK_HOURS = [(10, 17), (20, 22)] # 10 AM-5 PM and 8-10 PM
    # NIGHT: All other hours (10 PM - 7 AM)
    
    def __init__(self, model_path: str = None, algorithm: str = "PPO"):
        self.model_path = model_path or settings.MODEL_PATH
        self.algorithm = algorithm.upper()
        self.model = None
        self.loaded = False
        self.current_policy = None
        
        # Policy paths
        self.policy_paths = {
            'PEAK': settings.MODEL_POLICY_PEAK,
            'OFF_PEAK': settings.MODEL_POLICY_OFF_PEAK,
            'NIGHT': settings.MODEL_POLICY_NIGHT
        }
        
    def get_current_time_period(self) -> str:
        """
        Determine current time period based on hour of day
        
        Returns:
            str: 'PEAK', 'OFF_PEAK', or 'NIGHT'
        """
        hour = datetime.now().hour
        
        # Check if peak hours
        for start, end in self.PEAK_HOURS:
            if start <= hour < end:
                return 'PEAK'
        
        # Check if off-peak hours
        for start, end in self.OFF_PEAK_HOURS:
            if start <= hour < end:
                return 'OFF_PEAK'
        
        # Default to night
        return 'NIGHT'
    
    def get_policy_for_intensity(self, intensity: str = None) -> str:
        """
        Get policy based on traffic intensity parameter or current time
        
        Args:
            intensity: Optional intensity override ('peak', 'offpeak')
            
        Returns:
            str: Policy type ('PEAK', 'OFF_PEAK', 'NIGHT')
        """
        if intensity:
            if intensity.lower() == 'peak':
                return 'PEAK'
            elif intensity.lower() == 'offpeak':
                return 'OFF_PEAK'
        
        return self.get_current_time_period()
    
    def load_model(self, policy_type: str = None) -> bool:
        """
        Load trained RL model for specified policy type
        
        Args:
            policy_type: 'PEAK', 'OFF_PEAK', or 'NIGHT'. If None, auto-detect.
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Determine which policy to load
            if policy_type is None:
                policy_type = self.get_current_time_period()
            
            policy_type = policy_type.upper()
            
            # Check if already loaded with same policy
            if self.loaded and self.current_policy == policy_type:
                print(f"Policy {policy_type} already loaded")
                return True
            
            policy_path = self.policy_paths.get(policy_type)
            if not policy_path:
                print(f"Unknown policy type: {policy_type}")
                return False
            
            if not os.path.exists(policy_path):
                print(f"Model not found at {policy_path}")
                # Try fallback to any available model
                for ptype, ppath in self.policy_paths.items():
                    if os.path.exists(ppath):
                        print(f"Falling back to {ptype} policy at {ppath}")
                        policy_path = ppath
                        policy_type = ptype
                        break
                else:
                    print("No models found!")
                    return False
            
            print(f"Loading {self.algorithm} model ({policy_type}) from {policy_path}")
            
            if self.algorithm == "PPO":
                self.model = PPO.load(policy_path)
            elif self.algorithm == "DQN":
                self.model = DQN.load(policy_path)
            else:
                print(f"Unknown algorithm: {self.algorithm}")
                return False
            
            self.loaded = True
            self.current_policy = policy_type
            print(f"✓ Model loaded successfully: {policy_type} policy")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def switch_policy_if_needed(self, intensity: str = None) -> bool:
        """
        Check if policy needs to be switched based on time or intensity
        
        Args:
            intensity: Optional intensity override
            
        Returns:
            bool: True if policy was switched
        """
        required_policy = self.get_policy_for_intensity(intensity)
        
        if self.current_policy != required_policy:
            print(f"Switching policy from {self.current_policy} to {required_policy}")
            return self.load_model(required_policy)
        
        return False
    
    def predict_action(self, observation):
        """
        Predict action for given observation
        
        Args:
            observation: Current state observation
            
        Returns:
            Predicted action
        """
        if not self.loaded or self.model is None:
            print("Model not loaded")
            return None
        
        try:
            action, _states = self.model.predict(observation, deterministic=True)
            return action
            
        except Exception as e:
            print(f"Error predicting action: {e}")
            return None
    
    def get_observation_from_traci(self):
        """
        Get observation from current SUMO state via TraCI
        Matches the observation space expected by the trained model
        """
        if not traci_handler.connected:
            return None
        
        try:
            # Get all lane IDs from the simulation
            all_lanes = traci.lane.getIDList()
            
            # Filter for incoming lanes (exclude internal lanes starting with ':')
            incoming_lanes = [l for l in all_lanes if not l.startswith(':')][:8]
            
            # Pad if we have fewer than 8 lanes
            while len(incoming_lanes) < 8:
                incoming_lanes.append(incoming_lanes[-1] if incoming_lanes else "dummy")
            
            densities = []
            queues = []
            
            for lane_id in incoming_lanes:
                try:
                    # Density (occupancy normalized to 0-1)
                    occupancy = traci.lane.getLastStepOccupancy(lane_id) / 100.0
                    densities.append(min(1.0, max(0.0, occupancy)))
                    
                    # Queue (halting vehicles normalized)
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                    length = traci.lane.getLength(lane_id)
                    max_vehicles = max(1, length / 5.0)  # Assume 5m per vehicle
                    queues.append(min(1.0, queue / max_vehicles))
                except:
                    densities.append(0.0)
                    queues.append(0.0)
            
            # Current phase (normalized to 0-1 range for 4 phases)
            current_phase = 0
            if traci_handler.junction_ids:
                try:
                    phase = traci.trafficlight.getPhase(traci_handler.junction_ids[0])
                    current_phase = phase / 4.0  # Normalize assuming max 4 phases
                except:
                    current_phase = 0
            
            # Time of day (normalized 0-1)
            hour = datetime.now().hour
            time_normalized = hour / 24.0
            
            # Weather factor (default to clear weather = 1.0)
            weather_factor = 1.0
            
            # Combine: 8 densities + 8 queues + phase + time + weather = 19 features
            observation = densities + queues + [float(current_phase), time_normalized, weather_factor]
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None
    
    def control_traffic_light(self, junction_id: str, intensity: str = None):
        """
        Control traffic light using RL agent
        
        Args:
            junction_id: Traffic light junction ID
            intensity: Optional traffic intensity ('peak', 'offpeak')
            
        Returns:
            bool: True if action applied successfully
        """
        if not self.loaded:
            # Try to load model
            if not self.load_model(self.get_policy_for_intensity(intensity)):
                print("Failed to load model for control")
                return False
        
        # Switch policy if needed based on time/intensity
        self.switch_policy_if_needed(intensity)
        
        try:
            # Get current observation
            observation = self.get_observation_from_traci()
            if observation is None:
                return False
            
            # Predict action
            action = self.predict_action(observation)
            if action is None:
                return False
            
            # Apply action to traffic light
            success = traci_handler.set_traffic_light_phase(junction_id, int(action))
            
            return success
            
        except Exception as e:
            print(f"Error controlling traffic light: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get current agent status"""
        return {
            'loaded': self.loaded,
            'algorithm': self.algorithm,
            'current_policy': self.current_policy,
            'time_period': self.get_current_time_period(),
            'policy_paths': self.policy_paths
        }


# Global RL agent instance
rl_agent = RLAgent()
