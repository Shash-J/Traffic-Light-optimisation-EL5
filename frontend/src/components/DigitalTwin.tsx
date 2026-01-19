import React from 'react';
import { TrafficMap } from './TrafficMap';
import { SignalState } from './SignalState';
import { WeatherIndicator } from './WeatherIndicator';
import { EmergencyAlert } from './EmergencyAlert';

interface EmergencyVehicle {
    id: string;
    type: 'AMBULANCE' | 'FIRE_TRUCK' | 'POLICE';
    distance: number;
    eta: number;
    junction: string;
}

interface WeatherState {
    condition: number;  // 0-3 (normal, light rain, moderate rain, heavy rain)
    speedFactor: number;
    minGreenAdjustment: number;
}

interface DigitalTwinProps {
    queueLength: number;
    vehicleCount: number;
    trafficLights: any;
    locationId?: string;
    // New props for advanced features
    weatherState?: WeatherState;
    emergencyActive?: boolean;
    emergencyVehicle?: EmergencyVehicle;
    controllerType?: 'RL' | 'Fixed-Time' | 'Actuated';
}

const CONTROLLER_BADGES: Record<string, { icon: string; color: string }> = {
    'RL': { icon: '🤖', color: '#22c55e' },
    'Fixed-Time': { icon: '⏱️', color: '#64748b' },
    'Actuated': { icon: '📊', color: '#eab308' }
};

export const DigitalTwin: React.FC<DigitalTwinProps> = ({
    queueLength,
    vehicleCount,
    trafficLights,
    locationId = 'silk_board',
    weatherState,
    emergencyActive = false,
    emergencyVehicle,
    controllerType = 'RL'
}) => {
    // Determine congestion color
    const getCongestionColor = () => {
        if (queueLength > 80) return '#ef4444'; // Red
        if (queueLength > 40) return '#eab308'; // Yellow
        return '#22c55e'; // Green
    };

    const color = getCongestionColor();

    return (
        <div className="digital-twin-container" style={{
            width: '100%',
            height: '100%',
            position: 'relative',
            background: '#0f172a' // fallback background
        }}>
            {/* Integrated Traffic Map (Satellite/Schematic) */}
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1 }}>
                <TrafficMap
                    queueLength={queueLength}
                    vehicleCount={vehicleCount}
                    currentPhase={trafficLights?.current_phase || 0}
                    location={locationId}
                />
            </div>

            {/* HUD Overlay for Signal State (Floating Top Left) */}
            <div style={{ position: 'absolute', top: '20px', left: '20px', zIndex: 20, width: '300px' }}>
                <h3 style={{ margin: '0 0 10px 0', fontSize: '0.9em', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(0,0,0,0.6)', padding: '5px', borderRadius: '4px', width: 'fit-content' }}>
                    <span style={{ width: '8px', height: '8px', background: '#22c55e', borderRadius: '50%', boxShadow: '0 0 5px #22c55e' }}></span>
                    LIVE SIGNAL TELEMETRY
                </h3>
                <SignalState trafficLights={trafficLights} />
            </div>

            {/* Weather Indicator (Top Right) */}
            <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 20 }}>
                <WeatherIndicator
                    condition={weatherState?.condition || 0}
                    speedFactor={weatherState?.speedFactor || 1.0}
                    minGreenAdjustment={weatherState?.minGreenAdjustment || 0}
                />
            </div>

            {/* Emergency Alert (Bottom Left) */}
            <div style={{ position: 'absolute', bottom: '20px', left: '20px', zIndex: 20 }}>
                <EmergencyAlert
                    active={emergencyActive}
                    vehicle={emergencyVehicle}
                    junctionId={locationId}
                />
            </div>

            {/* Controller Type Badge (Top Center) */}
            <div style={{
                position: 'absolute',
                top: '20px',
                left: '50%',
                transform: 'translateX(-50%)',
                zIndex: 20,
                background: 'rgba(15, 23, 42, 0.9)',
                borderRadius: '20px',
                padding: '8px 16px',
                border: `1px solid ${CONTROLLER_BADGES[controllerType].color}40`,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
            }}>
                <span style={{ fontSize: '1.2em' }}>{CONTROLLER_BADGES[controllerType].icon}</span>
                <span style={{
                    color: CONTROLLER_BADGES[controllerType].color,
                    fontWeight: 'bold',
                    fontSize: '0.85em'
                }}>
                    {controllerType} Controller
                </span>
            </div>

            {/* Simulation Status (Bottom Right) */}
            <div style={{ position: 'absolute', bottom: '20px', right: '20px', zIndex: 20, textAlign: 'right' }}>
                <div style={{ fontSize: '0.8em', color: '#64748b', background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: '4px' }}>CONGESTION INDEX</div>
                <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: color, textShadow: '0 2px 4px rgba(0,0,0,0.5)' }}>
                    {queueLength > 80 ? 'CRITICAL' : (queueLength > 40 ? 'HEAVY' : 'STABLE')}
                </div>
                <div style={{ fontSize: '0.8em', color: '#94a3b8', marginTop: '5px' }}>
                    📍 {locationId.replace('_', ' ').toUpperCase()}
                </div>
            </div>

            {/* CSS Animations */}
            <style>{`
                @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                @keyframes pulse { 0% { transform: scale(1); border-color: ${color}; } 50% { transform: scale(1.05); border-color: white; } 100% { transform: scale(1); border-color: ${color}; } }
            `}</style>
        </div>
    );
};
