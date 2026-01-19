
import os
import sys

# Add backend to path so we can import app
# Current file is in backend/app/sumo/
# We want to add backend/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.sumo.network_manager import network_manager

def batch_convert():
    print("🚀 Starting Batch OSM Conversion...")
    
    osm_dir = network_manager.osm_dir
    converted_dir = network_manager.net_dir
    
    files = [f for f in os.listdir(osm_dir) if f.endswith('.osm')]
    
    if not files:
        print("⚠️ No .osm files found in", osm_dir)
        return

    success_count = 0
    
    for f in files:
        location_name = f.replace('.osm', '')
        osm_path = os.path.join(osm_dir, f)
        net_path = os.path.join(converted_dir, f.replace('.osm', '.net.xml'))
        
        print(f"\n📍 Processing: {location_name}")
        
        # Trigger conversion logic
        if network_manager._convert_osm_to_net(osm_path, net_path):
            print(f"   ✅ Converted: {f} -> {os.path.basename(net_path)}")
            
            # Also generate routes immediately
            route_path = os.path.join(converted_dir, f.replace('.osm', '.rou.xml'))
            if network_manager._generate_random_routes(net_path, route_path):
                 print(f"   ✅ Routes Generated: {os.path.basename(route_path)}")
            
            success_count += 1
        else:
            print(f"   ❌ Failed to convert: {f}")

    print(f"\n🎉 Batch processing complete! {success_count}/{len(files)} files processed.")

if __name__ == "__main__":
    batch_convert()
