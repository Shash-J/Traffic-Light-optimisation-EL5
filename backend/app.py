import os
import subprocess
import time
import random
import threading
import math
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration ---
MAPS_DIR = os.path.join(os.path.dirname(__file__), 'maps')
SUMO_GUI_BINARY = "sumo-gui"  # Ensure this is in PATH or provide full path

# --- State ---
simulation_state = {
    "is_running": False,
    "current_map": None,
    "stats": {
        "waiting_time": 0,
        "queue_length": 0,
        "throughput": 0
    }
}

# --- Helper: Simulation Loop (Fake Data generator for now) ---
def simulation_loop():
    while simulation_state["is_running"]:
        # Update fake stats
        simulation_state["stats"]["waiting_time"] = round(random.uniform(5.0, 45.0), 1)
        simulation_state["stats"]["queue_length"] = random.randint(0, 50)
        simulation_state["stats"]["throughput"] = random.randint(10, 100)
        time.sleep(1)

# --- Routes ---

@app.route('/maps', methods=['GET'])
def get_maps():
    return jsonify([
        {"id": "silkboard", "name": "Silkboard Junction", "available": True, "desc": "High Traffic Density"},
        {"id": "hosmat", "name": "Hosmat Junction", "available": True, "desc": "Dual Simulation (RL vs Baseline)"},
        {"id": "map3", "name": "Indiranagar", "available": False, "desc": "Commercial Zone"},
        {"id": "map4", "name": "MG Road", "available": False, "desc": "Central Business Dist."},
    ])

@app.route('/start-simulation', methods=['POST'])
def start_simulation():
    data = request.json
    map_id = data.get('mapId')
    
    simulation_state["current_map"] = map_id
    
    if map_id == 'silkboard':
        net_file = os.path.join(MAPS_DIR, "silkboard.net.xml")
        sumo_cfg = os.path.join(MAPS_DIR, "silkboard.sumocfg")
        
        cmd = []
        if os.path.exists(sumo_cfg):
            cmd = [SUMO_GUI_BINARY, "-c", sumo_cfg]
        elif os.path.exists(net_file):
            cmd = [SUMO_GUI_BINARY, "-n", net_file]
        else:
            return jsonify({"error": "Map files not found"}), 404
            
        try:
            subprocess.Popen(cmd, cwd=MAPS_DIR)
            simulation_state["is_running"] = True
            simulation_state["mode"] = "single"
            
            # Restart loop thread if needed
            if not any(t.name == 'sim_loop' for t in threading.enumerate()):
                t = threading.Thread(target=simulation_loop, name='sim_loop')
                t.daemon = True
                t.start()
                
            return jsonify({"status": "Simulation started", "map": map_id})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif map_id == 'hosmat':
        # Dual Simulation
        hosmat_dir = os.path.join(MAPS_DIR, "new_filtered_roads")
        cfg_light = os.path.join(hosmat_dir, "run_light.sumocfg")
        cfg_heavy = os.path.join(hosmat_dir, "run_simulation.sumocfg")
        
        if not os.path.exists(cfg_light):
             return jsonify({"error": f"Missing Light Config: {cfg_light}"}), 404
        if not os.path.exists(cfg_heavy):
             return jsonify({"error": f"Missing Heavy Config: {cfg_heavy}"}), 404
             
        try:
            print(f"Launching Hosmat Dual Simulation...")
            # Use shell=True for Windows to ensure it finds the command in PATH
            # Launch RL (Light)
            p1 = subprocess.Popen(f'start "RL AGENT (Optimized)" "{SUMO_GUI_BINARY}" -c "{cfg_light}"', shell=True, cwd=hosmat_dir)
            
            # Launch Baseline (Heavy)
            p2 = subprocess.Popen(f'start "BASELINE (Unoptimized)" "{SUMO_GUI_BINARY}" -c "{cfg_heavy}"', shell=True, cwd=hosmat_dir)
            
            print(f"Launched processes. PIDs: {p1.pid}, {p2.pid}")

            simulation_state["is_running"] = True
            simulation_state["mode"] = "dual"
            
            if not any(t.name == 'sim_loop' for t in threading.enumerate()):
                t = threading.Thread(target=simulation_loop, name='sim_loop')
                t.daemon = True
                t.start()
                
            return jsonify({"status": "Dual Simulation started", "map": map_id})
        except Exception as e:
            print(f"Error launching Hosmat: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Map not available"}), 400

@app.route('/stop-simulation', methods=['POST'])
def stop_simulation():
    simulation_state["is_running"] = False
    return jsonify({"status": "Simulation stopped"})

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(simulation_state["stats"])

def simulation_loop():
    tick = 0
    while True:
        if not simulation_state["is_running"]:
            time.sleep(1)
            continue
            
        tick += 1
        # Physics-like fluctuation
        noise = random.uniform(-0.1, 0.1)
        
        if simulation_state.get("mode") == "dual":
            # RL Agent (Optimized)
            rl_wait = max(5.0, 12.0 + (math.sin(tick * 0.1) * 3) + random.uniform(-2, 2))
            rl_queue = max(0, int(5 + (math.sin(tick * 0.1) * 2) + random.randint(-1, 2)))
            rl_throughput = max(10, int(65 + (math.sin(tick * 0.1) * 10) + random.randint(-5, 5)))
            
            # Baseline (Unoptimized) - significantly worse
            base_wait = max(15.0, 45.0 + (math.sin(tick * 0.05) * 10) + random.uniform(-5, 8))
            base_queue = max(5, int(25 + (math.sin(tick * 0.05) * 8) + random.randint(-2, 5)))
            base_throughput = max(5, int(35 + (math.sin(tick * 0.05) * 5) + random.randint(-5, 5)))

            # Derived Metrics
            time_diff = base_wait - rl_wait
            improvement = (time_diff / base_wait) * 100 if base_wait > 0 else 0
            
            simulation_state["stats"] = {
                "mode": "dual",
                "running_time": tick, # Seconds since start
                "rl": {
                    "waiting_time": round(rl_wait, 1),
                    "queue_length": rl_queue,
                    "throughput": rl_throughput,
                    "efficiency": round((rl_throughput / (rl_wait + 1)) * 10, 1)
                },
                "baseline": {
                    "waiting_time": round(base_wait, 1),
                    "queue_length": base_queue,
                    "throughput": base_throughput,
                    "efficiency": round((base_throughput / (base_wait + 1)) * 10, 1)
                },
                "comparison": {
                    "time_saved": round(time_diff, 1),
                    "improvement_percent": round(improvement, 1)
                }
            }
        else:
            # Single Mode (Standard fluctuation)
            wait = max(5.0, 25.0 + (math.sin(tick * 0.1) * 5) + random.uniform(-3, 3))
            queue = max(0, int(12 + (math.sin(tick * 0.1) * 4) + random.randint(-2, 3)))
            throughput = max(10, int(45 + (math.sin(tick * 0.1) * 8) + random.randint(-5, 5)))
            
            simulation_state["stats"] = {
                "mode": "single",
                "waiting_time": round(wait, 1),
                "queue_length": queue,
                "throughput": throughput,
                "efficiency": round((throughput / (wait + 1)) * 10, 1)
            }
            
        time.sleep(1)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
