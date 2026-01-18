import React, { useEffect, useState } from 'react';

interface EmergencyVehicle {
    id: string;
    type: 'AMBULANCE' | 'FIRE_TRUCK' | 'POLICE';
    distance: number;
    eta: number;
    junction: string;
}

interface EmergencyAlertProps {
    active: boolean;
    vehicle?: EmergencyVehicle;
    junctionId?: string;
}

const VEHICLE_ICONS: Record<string, string> = {
    'AMBULANCE': '🚑',
    'FIRE_TRUCK': '🚒',
    'POLICE': '🚓'
};

const VEHICLE_COLORS: Record<string, string> = {
    'AMBULANCE': '#ef4444',   // Red
    'FIRE_TRUCK': '#f97316',  // Orange
    'POLICE': '#3b82f6'       // Blue
};

export const EmergencyAlert: React.FC<EmergencyAlertProps> = ({ 
    active, 
    vehicle,
    junctionId 
}) => {
    const [blink, setBlink] = useState(false);

    useEffect(() => {
        if (active) {
            const interval = setInterval(() => {
                setBlink(b => !b);
            }, 500);
            return () => clearInterval(interval);
        }
    }, [active]);

    if (!active || !vehicle) {
        return (
            <div style={{
                background: 'rgba(15, 23, 42, 0.9)',
                borderRadius: '12px',
                padding: '16px',
                border: '1px solid #22c55e40',
                backdropFilter: 'blur(8px)',
                minWidth: '200px'
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px'
                }}>
                    <span style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        background: '#22c55e',
                        boxShadow: '0 0 8px #22c55e'
                    }}></span>
                    <div>
                        <div style={{ color: 'white', fontWeight: 'bold' }}>
                            No Emergency
                        </div>
                        <div style={{ fontSize: '0.75em', color: '#64748b' }}>
                            Normal operation
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    const icon = VEHICLE_ICONS[vehicle.type] || '🚨';
    const color = VEHICLE_COLORS[vehicle.type] || '#ef4444';

    return (
        <div style={{
            background: 'rgba(15, 23, 42, 0.95)',
            borderRadius: '12px',
            padding: '16px',
            border: `2px solid ${color}`,
            backdropFilter: 'blur(8px)',
            minWidth: '250px',
            animation: 'emergencyPulse 1s infinite',
            boxShadow: `0 0 20px ${color}50`
        }}>
            {/* Alert Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '12px'
            }}>
                <span style={{ 
                    fontSize: '2.5em',
                    animation: 'shake 0.5s infinite'
                }}>
                    {icon}
                </span>
                <div>
                    <div style={{
                        fontSize: '1.3em',
                        fontWeight: 'bold',
                        color: color,
                        textTransform: 'uppercase'
                    }}>
                        {vehicle.type.replace('_', ' ')}
                    </div>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px'
                    }}>
                        <span style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: blink ? color : 'transparent',
                            transition: 'background 0.2s'
                        }}></span>
                        <span style={{
                            fontSize: '0.8em',
                            color: '#f87171',
                            fontWeight: 'bold'
                        }}>
                            PRIORITY ACTIVE
                        </span>
                    </div>
                </div>
            </div>

            {/* Vehicle Info */}
            <div style={{
                background: 'rgba(0, 0, 0, 0.4)',
                borderRadius: '8px',
                padding: '12px',
                marginBottom: '12px'
            }}>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '10px'
                }}>
                    <div>
                        <div style={{ fontSize: '0.7em', color: '#64748b', marginBottom: '2px' }}>
                            Distance
                        </div>
                        <div style={{ fontSize: '1.4em', fontWeight: 'bold', color: 'white' }}>
                            {vehicle.distance.toFixed(0)}m
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.7em', color: '#64748b', marginBottom: '2px' }}>
                            ETA
                        </div>
                        <div style={{ fontSize: '1.4em', fontWeight: 'bold', color: 'white' }}>
                            {vehicle.eta.toFixed(1)}s
                        </div>
                    </div>
                </div>
            </div>

            {/* Action Status */}
            <div style={{
                background: `${color}20`,
                borderRadius: '6px',
                padding: '10px',
                textAlign: 'center'
            }}>
                <div style={{
                    fontSize: '0.85em',
                    color: 'white',
                    fontWeight: 'bold'
                }}>
                    🟢 GREEN LIGHT OVERRIDE
                </div>
                <div style={{
                    fontSize: '0.7em',
                    color: '#94a3b8',
                    marginTop: '4px'
                }}>
                    RL agent paused • Clearing path for emergency
                </div>
            </div>

            {/* ID */}
            <div style={{
                marginTop: '10px',
                fontSize: '0.7em',
                color: '#475569',
                textAlign: 'center'
            }}>
                Vehicle: {vehicle.id} | Junction: {junctionId || 'Unknown'}
            </div>

            <style>{`
                @keyframes emergencyPulse {
                    0% { box-shadow: 0 0 10px ${color}40; }
                    50% { box-shadow: 0 0 30px ${color}70; }
                    100% { box-shadow: 0 0 10px ${color}40; }
                }
                @keyframes shake {
                    0%, 100% { transform: translateX(0); }
                    25% { transform: translateX(-2px); }
                    75% { transform: translateX(2px); }
                }
            `}</style>
        </div>
    );
};

export default EmergencyAlert;
