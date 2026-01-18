import React from 'react';

interface WeatherIndicatorProps {
    condition: number;  // 0-3 (normal, light rain, moderate rain, heavy rain)
    speedFactor: number;
    minGreenAdjustment: number;
}

const WEATHER_ICONS: Record<number, string> = {
    0: '☀️',  // Normal
    1: '🌧️',  // Light rain
    2: '⛈️',  // Moderate rain
    3: '🌊'   // Heavy rain
};

const WEATHER_LABELS: Record<number, string> = {
    0: 'Clear',
    1: 'Light Rain',
    2: 'Moderate Rain',
    3: 'Heavy Rain'
};

const WEATHER_COLORS: Record<number, string> = {
    0: '#22c55e',  // Green
    1: '#60a5fa',  // Light blue
    2: '#3b82f6',  // Blue
    3: '#1d4ed8'   // Dark blue
};

export const WeatherIndicator: React.FC<WeatherIndicatorProps> = ({ 
    condition, 
    speedFactor, 
    minGreenAdjustment 
}) => {
    const icon = WEATHER_ICONS[condition] || WEATHER_ICONS[0];
    const label = WEATHER_LABELS[condition] || WEATHER_LABELS[0];
    const color = WEATHER_COLORS[condition] || WEATHER_COLORS[0];
    
    const isRaining = condition > 0;

    return (
        <div style={{
            background: 'rgba(15, 23, 42, 0.9)',
            borderRadius: '12px',
            padding: '16px',
            border: `1px solid ${color}40`,
            backdropFilter: 'blur(8px)',
            minWidth: '200px'
        }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                marginBottom: '12px'
            }}>
                <span style={{ fontSize: '2em' }}>{icon}</span>
                <div>
                    <div style={{ 
                        fontSize: '1.1em', 
                        fontWeight: 'bold',
                        color: 'white'
                    }}>
                        {label}
                    </div>
                    <div style={{
                        fontSize: '0.75em',
                        color: '#94a3b8',
                        textTransform: 'uppercase',
                        letterSpacing: '1px'
                    }}>
                        Weather Status
                    </div>
                </div>
            </div>
            
            {/* Metrics */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '10px',
                marginTop: '10px'
            }}>
                <div style={{
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '8px',
                    borderRadius: '6px'
                }}>
                    <div style={{ fontSize: '0.7em', color: '#64748b' }}>Speed Factor</div>
                    <div style={{ 
                        fontSize: '1.2em', 
                        fontWeight: 'bold',
                        color: speedFactor < 1 ? '#f59e0b' : '#22c55e'
                    }}>
                        {(speedFactor * 100).toFixed(0)}%
                    </div>
                </div>
                <div style={{
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '8px',
                    borderRadius: '6px'
                }}>
                    <div style={{ fontSize: '0.7em', color: '#64748b' }}>Min Green +</div>
                    <div style={{ 
                        fontSize: '1.2em', 
                        fontWeight: 'bold',
                        color: minGreenAdjustment > 0 ? '#f59e0b' : '#22c55e'
                    }}>
                        {minGreenAdjustment}s
                    </div>
                </div>
            </div>

            {/* Rain Animation */}
            {isRaining && (
                <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    overflow: 'hidden',
                    pointerEvents: 'none',
                    borderRadius: '12px'
                }}>
                    {[...Array(condition * 5)].map((_, i) => (
                        <div
                            key={i}
                            style={{
                                position: 'absolute',
                                width: '2px',
                                height: '10px',
                                background: 'linear-gradient(transparent, #60a5fa)',
                                left: `${Math.random() * 100}%`,
                                animation: `rain ${0.5 + Math.random() * 0.5}s linear infinite`,
                                animationDelay: `${Math.random() * 0.5}s`,
                                opacity: 0.3
                            }}
                        />
                    ))}
                </div>
            )}
            
            <style>{`
                @keyframes rain {
                    0% { transform: translateY(-10px); opacity: 0; }
                    50% { opacity: 0.3; }
                    100% { transform: translateY(100px); opacity: 0; }
                }
            `}</style>
        </div>
    );
};

export default WeatherIndicator;
