import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './Map.css';

// Fix Leaflet marker icon issue
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

interface RealWorldMapProps {
    location: string; // 'silk_board' | 'hebbal' | 'tin_factory'
    phase: number;    // 0-3
    queueLength: number;
}

const LOCATIONS = {
    silk_board: { lat: 12.9175, lng: 77.6235, zoom: 19 },
    hebbal: { lat: 13.0358, lng: 77.5970, zoom: 18 },
    tin_factory: { lat: 12.9961, lng: 77.6698, zoom: 18 }
};

// Component to handle map view updates
const MapUpdater: React.FC<{ center: [number, number], zoom: number }> = ({ center, zoom }) => {
    const map = useMap();
    useEffect(() => {
        map.flyTo(center, zoom, { duration: 1.5 });
    }, [center, zoom, map]);
    return null;
};

// Traffic Light Overlay Component
const TrafficLightOverlay: React.FC<{ phase: number, center: { lat: number, lng: number } }> = ({ phase, center }) => {
    // Define relative positions for lights based on center
    // Adjust these deltas to match the road layout of satellite view
    const offset = 0.0004;

    // 0: NS Green, 1: EW Green, 2: NS Right, 3: EW Right
    const getColor = (lightIdx: number) => {
        // Simple mapping: 
        // 0=North, 1=East, 2=South, 3=West
        // Phase 0: NS Green -> North(0) & South(2) Green
        // Phase 1: EW Green -> East(1) & West(3) Green

        const isGreen = (lightIdx === 0 || lightIdx === 2) ? (phase === 0 || phase === 2) : (phase === 1 || phase === 3);
        return isGreen ? '#00ff00' : '#ff0000';
    };

    const lights = [
        { id: 0, lat: center.lat + offset, lng: center.lng - 0.0001, label: 'N' },
        { id: 1, lat: center.lat + 0.0001, lng: center.lng + offset, label: 'E' },
        { id: 2, lat: center.lat - offset, lng: center.lng + 0.0001, label: 'S' },
        { id: 3, lat: center.lat - 0.0001, lng: center.lng - offset, label: 'W' }
    ];

    return (
        <>
            {lights.map((l, idx) => (
                <div
                    key={l.id}
                    style={{
                        position: 'absolute',
                        // We can't easily use absolute pixels on Leaflet without custom Pane or Markers with DivIcon
                        // So we use Markers with custom DivIcons for simplicity
                    }}
                />
            ))}
            {/* Using Leaflet Markers for Traffic Lights */}
            {lights.map((l, idx) => (
                <Marker
                    key={l.id}
                    position={[l.lat, l.lng]}
                    icon={L.divIcon({
                        className: 'custom-traffic-light',
                        html: `<div style="
                            background-color: ${getColor(idx)};
                            width: 20px;
                            height: 20px;
                            border-radius: 50%;
                            border: 2px solid white;
                            box-shadow: 0 0 10px ${getColor(idx)};
                        "></div>`,
                        iconSize: [20, 20],
                        iconAnchor: [10, 10]
                    })}
                />
            ))}
        </>
    );
};

export const RealWorldMap: React.FC<RealWorldMapProps> = ({ location, phase, queueLength }) => {
    const locKey = location.toLowerCase().replace(' ', '_') as keyof typeof LOCATIONS;
    const loc = LOCATIONS[locKey] || LOCATIONS.silk_board;

    return (
        <div className="relative w-full h-full rounded-lg overflow-hidden border border-gray-700 shadow-2xl">
            <MapContainer
                center={[loc.lat, loc.lng]}
                zoom={loc.zoom}
                style={{ height: '100%', width: '100%', background: '#0f0f1e' }}
                zoomControl={false}
                attributionControl={false}
            >
                {/* Dark Matter / Satellite Hybrid feel */}
                <TileLayer
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    attribution='&copy; video games'
                />

                {/* Dark overlay to make it look "command center" style */}
                <div style={{
                    position: 'absolute',
                    top: 0, left: 0, right: 0, bottom: 0,
                    pointerEvents: 'none',
                    zIndex: 400,
                    background: 'rgba(10, 15, 30, 0.4)' // Blue tint
                }}></div>

                <MapUpdater center={[loc.lat, loc.lng]} zoom={loc.zoom} />

                <TrafficLightOverlay phase={phase} center={loc} />
            </MapContainer>

            {/* Live Stats Overlay */}
            <div className="absolute top-4 right-4 bg-gray-900/90 backdrop-blur border border-cyan-500/30 p-4 rounded text-white z-[1000]">
                <h4 className="text-cyan-400 text-xs uppercase tracking-wider mb-2">Live Feed</h4>
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                    <span className="text-sm font-mono">LIVE</span>
                </div>
                <div className="text-xs text-gray-400 font-mono">
                    {loc.lat.toFixed(4)}, {loc.lng.toFixed(4)}
                </div>
            </div>
        </div>
    );
};
