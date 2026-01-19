"""
Network Manager
===============
Handles the management and conversion of SUMO networks from OSM files.
"""
import os
import subprocess
import glob
from typing import Optional
from app.config import settings

class NetworkManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.osm_dir = os.path.join(self.base_dir, "networks", "osm")
        self.net_dir = os.path.join(self.base_dir, "networks", "converted")
        
        # Ensure directories exist
        os.makedirs(self.osm_dir, exist_ok=True)
        os.makedirs(self.net_dir, exist_ok=True)
        
    def get_network_path(self, location: str) -> str:
        """
        Get the path to the .net.xml file for a given location.
        If only .osm exists, it attempts to convert it first.
        """
        # Normalize location name (e.g. "Silk Board" -> "silk_board")
        loc_id = location.lower().replace(" ", "_")
        
        net_file = os.path.join(self.net_dir, f"{loc_id}.net.xml")
        osm_file = os.path.join(self.osm_dir, f"{loc_id}.osm")
        
        # Check if converted network exists
        if os.path.exists(net_file):
            return net_file
            
        # Check if OSM file exists and convert
        if os.path.exists(osm_file):
            print(f"🔄 Converting OSM to SUMO network for {location}...")
            if self._convert_osm_to_net(osm_file, net_file):
                return net_file
        
        # Fallback to default simulation config if specific net not found
        # This prevents crash if file is missing
        print(f"⚠️ Network for {location} not found. Using default.")
        return os.path.join(self.base_dir, "network", "simulation.sumocfg") # Return config or None?

    
    def get_routes_path(self, location: str, net_file: str) -> str:
        """
        Get path to route file. Generates random traffic if not found.
        """
        loc_id = location.lower().replace(" ", "_")
        route_file = os.path.join(self.net_dir, f"{loc_id}.rou.xml")
        
        if os.path.exists(route_file):
            return route_file
            
        print(f"🔄 Generating random routes for {location}...")
        if self._generate_random_routes(net_file, route_file):
            return route_file
            
        return ""

    def _generate_random_routes(self, net_path: str, output_path: str) -> bool:
        """
        Generate random traffic using sumo/tools/randomTrips.py
        """
        try:
            # Find randomTrips.py
            tools_dir = os.path.join(settings.SUMO_HOME, "tools")
            random_trips = os.path.join(tools_dir, "randomTrips.py")
            
            if not os.path.exists(random_trips):
                print(f"❌ randomTrips.py not found at {random_trips}")
                return False
                
            # randomTrips.py -n net.xml -o trips.xml -r routes.rou.xml -e 3600 -p 1.0
            cmd = [
                "python", random_trips,
                "-n", net_path,
                "-r", output_path,
                "-e", "3600", # End time (1 hour)
                "-p", "2.0",  # Period (arrival rate, 1 veh every 2s)
                "--allow-fringe"
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"   ✅ Routes generated: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Route generation error: {e}")
            return False

    def _convert_osm_to_net(self, osm_path: str, net_path: str) -> bool:
        """
        Convert .osm to .net.xml using netconvert
        """
        try:
            # Construct netconvert command
            # netconvert --osm-files silk_board.osm --output-file silk_board.net.xml --geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
            
            cmd = [
                "netconvert",
                "--osm-files", osm_path,
                "-o", net_path,
                "--geometry.remove", "true",
                "--roundabouts.guess", "true",
                "--ramps.guess", "true",
                "--junctions.join", "true",
                "--tls.guess-signals", "true",
                "--tls.discard-simple", "true",
                "--tls.join", "true",
                "--no-warnings"
            ]
            
            # If explicit SUMO_HOME set, might need to use full path to binary?
            # Usually netconvert is in path if SUMO is installed.
            # If not, try to find it in SUMO_HOME
            if settings.SUMO_HOME:
                 bin_path = os.path.join(settings.SUMO_HOME, "bin", "netconvert")
                 if os.path.exists(bin_path) or os.path.exists(bin_path + ".exe"):
                     cmd[0] = bin_path

            print(f"   Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"   ✅ Conversion successful: {os.path.basename(net_path)}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ netconvert failed: {e}")
            print(f"   Stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return False

# Global instance
network_manager = NetworkManager()
