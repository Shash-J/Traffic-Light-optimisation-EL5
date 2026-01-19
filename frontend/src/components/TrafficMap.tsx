/**
 * Traffic Map Component
 * Visual representation of the intersection
 */
import React, { useState } from 'react';
import { RealWorldMap } from './RealWorldMap';

interface TrafficMapProps {
    queueLength: number;
    vehicleCount: number;
    currentPhase?: number; // Added phase prop
    location?: string;     // Added location prop
}

export const TrafficMap: React.FC<TrafficMapProps> = ({
    queueLength,
    vehicleCount,
    currentPhase = 0,
    location = "silk_board"
}) => {
    const [viewMode, setViewMode] = useState<'schematic' | 'real'>('real');

    return (
        <div className="traffic-map-container relative h-full">
            <div className="flex justify-between items-center mb-4">
                <h3>Intersection Overview</h3>
                <div className="flex bg-gray-800 rounded p-1">
                    <button
                        className={`px-3 py-1 text-xs rounded ${viewMode === 'real' ? 'bg-cyan-600 text-white' : 'text-gray-400 hover:text-white'}`}
                        onClick={() => setViewMode('real')}
                    >
                        Satellite
                    </button>
                    <button
                        className={`px-3 py-1 text-xs rounded ${viewMode === 'schematic' ? 'bg-cyan-600 text-white' : 'text-gray-400 hover:text-white'}`}
                        onClick={() => setViewMode('schematic')}
                    >
                        Schematic
                    </button>
                </div>
            </div>

            <div className="map-wrapper h-[400px] w-full relative">
                {viewMode === 'real' ? (
                    <RealWorldMap
                        location={location}
                        phase={currentPhase}
                        queueLength={queueLength}
                    />
                ) : (
                    <svg viewBox="0 0 400 400" className="intersection-svg w-full h-full">
                        {/* Background */}
                        <rect width="400" height="400" fill="#0f0f1e" />

                        {/* Roads */}
                        {/* Vertical road */}
                        <rect x="150" y="0" width="100" height="400" fill="#2a2a3e" />
                        {/* Horizontal road */}
                        <rect x="0" y="150" width="400" height="100" fill="#2a2a3e" />

                        {/* Lane markings */}
                        {/* Vertical center line */}
                        <line x1="200" y1="0" x2="200" y2="150" stroke="#ffff00" strokeWidth="2" strokeDasharray="10,10" />
                        <line x1="200" y1="250" x2="200" y2="400" stroke="#ffff00" strokeWidth="2" strokeDasharray="10,10" />
                        {/* Horizontal center line */}
                        <line x1="0" y1="200" x2="150" y2="200" stroke="#ffff00" strokeWidth="2" strokeDasharray="10,10" />
                        <line x1="250" y1="200" x2="400" y2="200" stroke="#ffff00" strokeWidth="2" strokeDasharray="10,10" />

                        {/* Junction box */}
                        <rect x="150" y="150" width="100" height="100" fill="#1a1a2e" stroke="#ffff00" strokeWidth="2" />

                        {/* Traffic lights */}
                        {/* North */}
                        <circle cx="180" cy="140" r="8" fill={currentPhase === 0 || currentPhase === 2 ? "#00ff00" : "#ff0000"} className="traffic-light" />
                        <circle cx="220" cy="140" r="8" fill={currentPhase === 0 || currentPhase === 2 ? "#00ff00" : "#ff0000"} className="traffic-light" />

                        {/* South */}
                        <circle cx="180" cy="260" r="8" fill={currentPhase === 0 || currentPhase === 2 ? "#00ff00" : "#ff0000"} className="traffic-light" />
                        <circle cx="220" cy="260" r="8" fill={currentPhase === 0 || currentPhase === 2 ? "#00ff00" : "#ff0000"} className="traffic-light" />

                        {/* East */}
                        <circle cx="260" cy="180" r="8" fill={currentPhase === 1 || currentPhase === 3 ? "#00ff00" : "#ff0000"} className="traffic-light" />
                        <circle cx="260" cy="220" r="8" fill={currentPhase === 1 || currentPhase === 3 ? "#00ff00" : "#ff0000"} className="traffic-light" />

                        {/* West */}
                        <circle cx="140" cy="180" r="8" fill={currentPhase === 1 || currentPhase === 3 ? "#00ff00" : "#ff0000"} className="traffic-light" />
                        <circle cx="140" cy="220" r="8" fill={currentPhase === 1 || currentPhase === 3 ? "#00ff00" : "#ff0000"} className="traffic-light" />

                        {/* Direction arrows */}
                        <text x="200" y="80" textAnchor="middle" fill="#fff" fontSize="20">↑</text>
                        <text x="200" y="100" textAnchor="middle" fill="#aaa" fontSize="12">N</text>
                        <text x="200" y="320" textAnchor="middle" fill="#fff" fontSize="20">↓</text>
                        <text x="200" y="340" textAnchor="middle" fill="#aaa" fontSize="12">S</text>
                        <text x="320" y="205" textAnchor="middle" fill="#fff" fontSize="20">→</text>
                        <text x="340" y="205" textAnchor="middle" fill="#aaa" fontSize="12">E</text>
                        <text x="80" y="205" textAnchor="middle" fill="#fff" fontSize="20">←</text>
                        <text x="60" y="205" textAnchor="middle" fill="#aaa" fontSize="12">W</text>
                    </svg>
                )}
            </div>

            <div className="map-stats mt-4 grid grid-cols-2 gap-4">
                <div className="stat-item bg-gray-800 p-3 rounded">
                    <span className="stat-label block text-gray-400 text-xs">Queue Length</span>
                    <span className="stat-value text-xl font-bold text-cyan-400">{queueLength}</span>
                </div>
                <div className="stat-item bg-gray-800 p-3 rounded">
                    <span className="stat-label block text-gray-400 text-xs">Active Vehicles</span>
                    <span className="stat-value text-xl font-bold text-cyan-400">{vehicleCount}</span>
                </div>
            </div>
        </div>
    );
};
