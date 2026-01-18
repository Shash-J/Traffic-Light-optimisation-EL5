"""
Generate valid routes for real network files
Uses SUMO's randomTrips to create routes that match actual edge IDs
"""
import subprocess
import os
from pathlib import Path

NETWORK_DIR = Path(__file__).parent
SUMO_HOME = os.environ.get('SUMO_HOME', 'C:/Program Files (x86)/Eclipse/Sumo')
RANDOM_TRIPS = Path(SUMO_HOME) / 'tools' / 'randomTrips.py'

# Networks to generate routes for
NETWORKS = {
    'silk_board': {
        'net': 'silk_board.net.xml',
        'route': 'routes_silk_board.rou.xml',
        'period': 7  # ~520 veh/hour (peak traffic)
    },
    'tin_factory': {
        'net': 'tin_factory.net.xml',
        'route': 'routes_tin_factory.rou.xml',
        'period': 7
    },
    'hebbal': {
        'net': 'hebbal.net.xml',
        'route': 'routes_hebbal.rou.xml',
        'period': 7
    },
    'network': {
        'net': 'network.net.xml',
        'route': 'routes_peak.rou.xml',
        'period': 7
    }
}

def generate_routes(net_name, config):
    net_file = NETWORK_DIR / config['net']
    route_file = NETWORK_DIR / config['route']
    
    if not net_file.exists():
        print(f"⚠️  Network file not found: {net_file}")
        return False
    
    print(f"\n🔧 Generating routes for {net_name}...")
    
    cmd = [
        'python', str(RANDOM_TRIPS),
        '-n', str(net_file),
        '-r', str(route_file),
        '-b', '0',
        '-e', '3600',
        '-p', str(config['period']),
        '--fringe-factor', '10',
        '--min-distance', '50',
        '--trip-attributes', 'departLane="best" departSpeed="max"',
        '--validate'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NETWORK_DIR))
        if result.returncode == 0:
            print(f"✅ Routes generated: {route_file.name}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚦 SUMO Route Generator")
    print("=" * 60)
    
    if not RANDOM_TRIPS.exists():
        print(f"❌ randomTrips.py not found at: {RANDOM_TRIPS}")
        print(f"   Please check SUMO_HOME: {SUMO_HOME}")
        exit(1)
    
    success_count = 0
    for net_name, config in NETWORKS.items():
        if generate_routes(net_name, config):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Generated {success_count}/{len(NETWORKS)} route files successfully!")
    print("=" * 60)
