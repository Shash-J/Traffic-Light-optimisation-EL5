"""
Route Generator with Emergency Vehicles
Generates SUMO route files with regular vehicles and emergency vehicles

Emergency vehicle types:
- Ambulance (highest priority)
- Fire truck (high priority)  
- Police (medium priority)

Based on Bangalore traffic patterns:
- Morning peak: 8-10 AM
- Evening peak: 5-7 PM
- Emergency injection: Random throughout simulation
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class VehicleTypeConfig:
    """Configuration for a vehicle type"""
    id: str
    accel: float = 2.6
    decel: float = 4.5
    max_speed: float = 13.89  # 50 km/h
    length: float = 5.0
    vclass: str = "passenger"
    priority: int = 0
    color: str = "1,1,0"  # Yellow default


# Vehicle type configurations
VEHICLE_TYPES = {
    'car': VehicleTypeConfig(
        id='car',
        accel=2.6,
        decel=4.5,
        max_speed=13.89,
        length=5.0,
        vclass='passenger',
        priority=0,
        color='1,1,0'
    ),
    'bus': VehicleTypeConfig(
        id='bus',
        accel=1.2,
        decel=4.0,
        max_speed=11.11,  # 40 km/h
        length=12.0,
        vclass='bus',
        priority=0,
        color='0,0.5,1'
    ),
    'bike': VehicleTypeConfig(
        id='bike',
        accel=3.0,
        decel=3.0,
        max_speed=8.33,  # 30 km/h
        length=2.0,
        vclass='bicycle',
        priority=0,
        color='0,1,0'
    ),
    'ambulance': VehicleTypeConfig(
        id='ambulance',
        accel=2.8,
        decel=4.5,
        max_speed=25.0,  # 90 km/h
        length=6.0,
        vclass='emergency',
        priority=5,
        color='1,0,0'  # Red
    ),
    'fire_truck': VehicleTypeConfig(
        id='fire_truck',
        accel=2.0,
        decel=4.0,
        max_speed=22.22,  # 80 km/h
        length=8.0,
        vclass='emergency',
        priority=4,
        color='1,0.3,0'  # Orange-red
    ),
    'police': VehicleTypeConfig(
        id='police',
        accel=3.0,
        decel=5.0,
        max_speed=27.78,  # 100 km/h
        length=5.0,
        vclass='emergency',
        priority=3,
        color='0,0,1'  # Blue
    )
}


def generate_vehicle_types_xml() -> str:
    """Generate vehicle types XML section"""
    lines = []
    
    for vtype in VEHICLE_TYPES.values():
        lines.append(f'''    <vType id="{vtype.id}" 
           accel="{vtype.accel}" 
           decel="{vtype.decel}" 
           maxSpeed="{vtype.max_speed}"
           length="{vtype.length}"
           vClass="{vtype.vclass}"
           color="{vtype.color}"/>''')
           
    return '\n'.join(lines)


def generate_routes_with_emergency(
    network_file: str,
    output_file: str,
    duration: int = 3600,
    demand_rate: float = 0.15,  # vehicles per second
    emergency_rate: float = 0.002,  # emergency vehicles per second
    peak_hours: List[Tuple[int, int]] = None,
    seed: int = 42
):
    """
    Generate route file with regular and emergency vehicles
    
    Args:
        network_file: Path to SUMO network file
        output_file: Output route file path
        duration: Simulation duration in seconds
        demand_rate: Base vehicle generation rate
        emergency_rate: Emergency vehicle generation rate
        peak_hours: List of (start, end) tuples for peak hours
        seed: Random seed
    """
    random.seed(seed)
    
    if peak_hours is None:
        # Default peak hours (scaled to simulation time)
        # Morning peak: 20-30% of simulation
        # Evening peak: 60-70% of simulation
        peak_hours = [
            (int(duration * 0.2), int(duration * 0.3)),
            (int(duration * 0.6), int(duration * 0.7))
        ]
    
    # Define routes (edges) - these should match your network
    # For a simple 4-way intersection:
    routes = [
        ("route_ns", ["south_in", "south_to_center", "center_to_north", "north_out"]),
        ("route_sn", ["north_in", "north_to_center", "center_to_south", "south_out"]),
        ("route_ew", ["west_in", "west_to_center", "center_to_east", "east_out"]),
        ("route_we", ["east_in", "east_to_center", "center_to_west", "west_out"]),
    ]
    
    # Emergency routes (typically hospital to accident locations)
    emergency_routes = [
        ("emerg_ns", ["south_in", "south_to_center", "center_to_north", "north_out"]),
        ("emerg_sn", ["north_in", "north_to_center", "center_to_south", "south_out"]),
        ("emerg_ew", ["west_in", "west_to_center", "center_to_east", "east_out"]),
        ("emerg_we", ["east_in", "east_to_center", "center_to_west", "west_out"]),
    ]
    
    # Generate XML
    xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_content.append(f'''
<!-- 
Generated route file with emergency vehicles
Duration: {duration}s
Demand rate: {demand_rate} veh/s
Emergency rate: {emergency_rate} veh/s
-->
''')
    xml_content.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    
    # Add vehicle types
    xml_content.append('\n    <!-- Vehicle Types -->')
    xml_content.append(generate_vehicle_types_xml())
    
    # Add route definitions
    xml_content.append('\n    <!-- Route Definitions -->')
    for route_id, edges in routes + emergency_routes:
        xml_content.append(f'    <route id="{route_id}" edges="{" ".join(edges)}"/>')
    
    # Generate vehicles
    xml_content.append('\n    <!-- Regular Vehicles -->')
    
    vehicle_id = 0
    current_time = 0
    
    while current_time < duration:
        # Check if in peak hour
        is_peak = any(start <= current_time <= end for start, end in peak_hours)
        current_rate = demand_rate * (2.0 if is_peak else 1.0)
        
        # Generate regular vehicles
        if random.random() < current_rate:
            route_id, _ = random.choice(routes)
            vehicle_type = random.choice(['car', 'car', 'car', 'bus', 'bike'])  # 60% cars
            
            xml_content.append(f'    <vehicle id="veh_{vehicle_id}" type="{vehicle_type}" route="{route_id}" depart="{current_time:.2f}" departLane="best" departSpeed="max"/>')
            vehicle_id += 1
        
        current_time += 1
    
    # Generate emergency vehicles
    xml_content.append('\n    <!-- Emergency Vehicles -->')
    
    emergency_id = 0
    current_time = 0
    
    while current_time < duration:
        if random.random() < emergency_rate:
            route_id, _ = random.choice(emergency_routes)
            emergency_type = random.choices(
                ['ambulance', 'fire_truck', 'police'],
                weights=[0.6, 0.2, 0.2]  # 60% ambulance
            )[0]
            
            xml_content.append(f'    <vehicle id="emerg_{emergency_id}" type="{emergency_type}" route="{route_id}" depart="{current_time:.2f}" departLane="best" departSpeed="max"/>')
            emergency_id += 1
            
        current_time += 1
    
    xml_content.append('\n</routes>')
    
    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(xml_content))
    
    print(f"Generated route file: {output_file}")
    print(f"  - Regular vehicles: {vehicle_id}")
    print(f"  - Emergency vehicles: {emergency_id}")
    
    return {
        'total_vehicles': vehicle_id,
        'emergency_vehicles': emergency_id,
        'output_file': output_file
    }


def generate_bangalore_scenarios(output_dir: str):
    """
    Generate multiple scenario route files for Bangalore
    
    Scenarios:
    1. Normal traffic (off-peak)
    2. Morning peak
    3. Evening peak
    4. Heavy emergency day (accidents)
    5. Monsoon conditions (reduced speeds, more vehicles)
    """
    scenarios = [
        {
            'name': 'normal_offpeak',
            'duration': 3600,
            'demand_rate': 0.1,
            'emergency_rate': 0.001,
            'peak_hours': []
        },
        {
            'name': 'morning_peak',
            'duration': 3600,
            'demand_rate': 0.15,
            'emergency_rate': 0.002,
            'peak_hours': [(0, 3600)]  # Entire simulation is peak
        },
        {
            'name': 'evening_peak',
            'duration': 3600,
            'demand_rate': 0.18,
            'emergency_rate': 0.002,
            'peak_hours': [(0, 3600)]
        },
        {
            'name': 'high_emergency',
            'duration': 3600,
            'demand_rate': 0.12,
            'emergency_rate': 0.01,  # 5x more emergencies
            'peak_hours': []
        },
        {
            'name': 'monsoon_heavy',
            'duration': 3600,
            'demand_rate': 0.2,  # More vehicles stuck
            'emergency_rate': 0.005,  # More accidents
            'peak_hours': [(0, 3600)]
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        output_file = os.path.join(output_dir, f"routes_{scenario['name']}.rou.xml")
        
        result = generate_routes_with_emergency(
            network_file="",  # Not needed for generation
            output_file=output_file,
            duration=scenario['duration'],
            demand_rate=scenario['demand_rate'],
            emergency_rate=scenario['emergency_rate'],
            peak_hours=scenario['peak_hours']
        )
        
        results[scenario['name']] = result
        
    return results


def add_emergency_to_existing_routes(
    input_file: str,
    output_file: str,
    emergency_rate: float = 0.002,
    seed: int = 42
):
    """
    Add emergency vehicles to an existing route file
    
    Args:
        input_file: Existing route file
        output_file: Output file path
        emergency_rate: Emergency vehicle generation rate
        seed: Random seed
    """
    random.seed(seed)
    
    # Parse existing file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Find max depart time and routes
    max_depart = 0
    routes = set()
    
    for vehicle in root.findall('.//vehicle'):
        depart = float(vehicle.get('depart', 0))
        max_depart = max(max_depart, depart)
        
        route_elem = vehicle.find('route')
        if route_elem is not None:
            edges = route_elem.get('edges', '').split()
            if edges:
                routes.add(tuple(edges))
    
    # Add vehicle types if not present
    existing_types = {vt.get('id') for vt in root.findall('.//vType')}
    
    for type_id, vtype in VEHICLE_TYPES.items():
        if type_id not in existing_types:
            vtype_elem = ET.SubElement(root, 'vType')
            vtype_elem.set('id', vtype.id)
            vtype_elem.set('accel', str(vtype.accel))
            vtype_elem.set('decel', str(vtype.decel))
            vtype_elem.set('maxSpeed', str(vtype.max_speed))
            vtype_elem.set('length', str(vtype.length))
            vtype_elem.set('vClass', vtype.vclass)
            vtype_elem.set('color', vtype.color)
    
    # Generate emergency vehicles
    emergency_id = 0
    current_time = 0
    routes_list = list(routes)
    
    while current_time <= max_depart:
        if random.random() < emergency_rate:
            emergency_type = random.choices(
                ['ambulance', 'fire_truck', 'police'],
                weights=[0.6, 0.2, 0.2]
            )[0]
            
            edges = random.choice(routes_list)
            
            vehicle = ET.SubElement(root, 'vehicle')
            vehicle.set('id', f'emerg_{emergency_id}')
            vehicle.set('type', emergency_type)
            vehicle.set('depart', f'{current_time:.2f}')
            vehicle.set('departLane', 'best')
            vehicle.set('departSpeed', 'max')
            
            route = ET.SubElement(vehicle, 'route')
            route.set('edges', ' '.join(edges))
            
            emergency_id += 1
            
        current_time += 1
    
    # Write output
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    
    print(f"Added {emergency_id} emergency vehicles to {output_file}")
    
    return emergency_id


if __name__ == "__main__":
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    network_dir = script_dir
    
    # Generate routes with emergency for network.net.xml
    generate_routes_with_emergency(
        network_file=os.path.join(network_dir, "network.net.xml"),
        output_file=os.path.join(network_dir, "routes_with_emergency.rou.xml"),
        duration=3600,
        demand_rate=0.15,
        emergency_rate=0.003
    )
    
    print("\n✅ Route generation complete!")
