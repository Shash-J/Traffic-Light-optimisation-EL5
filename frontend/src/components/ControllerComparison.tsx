import React from 'react';

interface ComparisonData {
    controller: string;
    avgDelay: number;
    throughput: number;
    emergencyTime: number;
    queueLength: number;
}

interface ControllerComparisonProps {
    data: ComparisonData[];
    improvement?: {
        delayReduction: number;
        throughputGain: number;
    };
}

const CONTROLLER_COLORS: Record<string, string> = {
    'Fixed-Time': '#64748b',
    'Actuated': '#eab308',
    'RL': '#22c55e'
};

const CONTROLLER_ICONS: Record<string, string> = {
    'Fixed-Time': '⏱️',
    'Actuated': '📊',
    'RL': '🤖'
};

export const ControllerComparison: React.FC<ControllerComparisonProps> = ({ 
    data, 
    improvement 
}) => {
    const maxDelay = Math.max(...data.map(d => d.avgDelay));
    const maxThroughput = Math.max(...data.map(d => d.throughput));

    return (
        <div style={{
            background: 'rgba(15, 23, 42, 0.95)',
            borderRadius: '16px',
            padding: '24px',
            border: '1px solid rgba(100, 116, 139, 0.3)'
        }}>
            {/* Header */}
            <div style={{
                marginBottom: '24px',
                borderBottom: '1px solid rgba(100, 116, 139, 0.2)',
                paddingBottom: '16px'
            }}>
                <h3 style={{
                    margin: 0,
                    color: 'white',
                    fontSize: '1.3em',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px'
                }}>
                    📈 Controller Performance Comparison
                </h3>
                <p style={{
                    margin: '8px 0 0 0',
                    color: '#64748b',
                    fontSize: '0.85em'
                }}>
                    Comparing Fixed-Time, Actuated, and RL controllers
                </p>
            </div>

            {/* Controller Cards */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '16px',
                marginBottom: '24px'
            }}>
                {data.map((controller) => (
                    <div
                        key={controller.controller}
                        style={{
                            background: 'rgba(0, 0, 0, 0.3)',
                            borderRadius: '12px',
                            padding: '16px',
                            border: `1px solid ${CONTROLLER_COLORS[controller.controller]}40`,
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        {/* Best badge for RL */}
                        {controller.controller === 'RL' && (
                            <div style={{
                                position: 'absolute',
                                top: '8px',
                                right: '8px',
                                background: '#22c55e',
                                color: 'black',
                                padding: '2px 8px',
                                borderRadius: '4px',
                                fontSize: '0.65em',
                                fontWeight: 'bold'
                            }}>
                                BEST
                            </div>
                        )}

                        {/* Controller Name */}
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: '16px'
                        }}>
                            <span style={{ fontSize: '1.5em' }}>
                                {CONTROLLER_ICONS[controller.controller]}
                            </span>
                            <div style={{
                                color: CONTROLLER_COLORS[controller.controller],
                                fontWeight: 'bold',
                                fontSize: '1.1em'
                            }}>
                                {controller.controller}
                            </div>
                        </div>

                        {/* Metrics */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {/* Delay */}
                            <div>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    fontSize: '0.75em',
                                    color: '#94a3b8',
                                    marginBottom: '4px'
                                }}>
                                    <span>Avg Delay</span>
                                    <span style={{ color: 'white', fontWeight: 'bold' }}>
                                        {controller.avgDelay.toFixed(1)}s
                                    </span>
                                </div>
                                <div style={{
                                    height: '6px',
                                    background: 'rgba(0, 0, 0, 0.3)',
                                    borderRadius: '3px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        height: '100%',
                                        width: `${(controller.avgDelay / maxDelay) * 100}%`,
                                        background: controller.avgDelay === Math.min(...data.map(d => d.avgDelay))
                                            ? '#22c55e' : '#64748b',
                                        borderRadius: '3px',
                                        transition: 'width 0.5s'
                                    }}></div>
                                </div>
                            </div>

                            {/* Throughput */}
                            <div>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    fontSize: '0.75em',
                                    color: '#94a3b8',
                                    marginBottom: '4px'
                                }}>
                                    <span>Throughput</span>
                                    <span style={{ color: 'white', fontWeight: 'bold' }}>
                                        {controller.throughput.toFixed(0)}/hr
                                    </span>
                                </div>
                                <div style={{
                                    height: '6px',
                                    background: 'rgba(0, 0, 0, 0.3)',
                                    borderRadius: '3px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        height: '100%',
                                        width: `${(controller.throughput / maxThroughput) * 100}%`,
                                        background: controller.throughput === maxThroughput
                                            ? '#22c55e' : '#64748b',
                                        borderRadius: '3px',
                                        transition: 'width 0.5s'
                                    }}></div>
                                </div>
                            </div>

                            {/* Emergency Time */}
                            <div>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    fontSize: '0.75em',
                                    color: '#94a3b8',
                                    marginBottom: '4px'
                                }}>
                                    <span>🚨 Emergency</span>
                                    <span style={{ 
                                        color: controller.emergencyTime < 15 ? '#22c55e' : '#f59e0b',
                                        fontWeight: 'bold' 
                                    }}>
                                        {controller.emergencyTime.toFixed(1)}s
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Improvement Summary */}
            {improvement && (
                <div style={{
                    background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05))',
                    borderRadius: '12px',
                    padding: '16px',
                    border: '1px solid rgba(34, 197, 94, 0.3)'
                }}>
                    <div style={{
                        fontSize: '0.85em',
                        color: '#94a3b8',
                        marginBottom: '12px',
                        textTransform: 'uppercase',
                        letterSpacing: '1px'
                    }}>
                        RL Improvement vs Fixed-Time
                    </div>
                    <div style={{
                        display: 'flex',
                        gap: '24px'
                    }}>
                        <div>
                            <div style={{
                                fontSize: '2em',
                                fontWeight: 'bold',
                                color: '#22c55e'
                            }}>
                                ↓ {improvement.delayReduction.toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.8em', color: '#64748b' }}>
                                Delay Reduction
                            </div>
                        </div>
                        <div>
                            <div style={{
                                fontSize: '2em',
                                fontWeight: 'bold',
                                color: '#22c55e'
                            }}>
                                ↑ {improvement.throughputGain.toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.8em', color: '#64748b' }}>
                                Throughput Gain
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Features Comparison Table */}
            <div style={{
                marginTop: '24px',
                overflowX: 'auto'
            }}>
                <table style={{
                    width: '100%',
                    borderCollapse: 'collapse',
                    fontSize: '0.85em'
                }}>
                    <thead>
                        <tr>
                            <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', borderBottom: '1px solid #334155' }}>
                                Feature
                            </th>
                            <th style={{ textAlign: 'center', padding: '8px', color: '#64748b', borderBottom: '1px solid #334155' }}>
                                Fixed-Time
                            </th>
                            <th style={{ textAlign: 'center', padding: '8px', color: '#64748b', borderBottom: '1px solid #334155' }}>
                                Actuated
                            </th>
                            <th style={{ textAlign: 'center', padding: '8px', color: '#64748b', borderBottom: '1px solid #334155' }}>
                                RL (Ours)
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style={{ padding: '8px', color: 'white' }}>Adaptive Timing</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>⚠️</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>✅</td>
                        </tr>
                        <tr>
                            <td style={{ padding: '8px', color: 'white' }}>Emergency Priority</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>✅</td>
                        </tr>
                        <tr>
                            <td style={{ padding: '8px', color: 'white' }}>Weather Awareness</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>✅</td>
                        </tr>
                        <tr>
                            <td style={{ padding: '8px', color: 'white' }}>Multi-Junction Coord.</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>❌</td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>✅</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default ControllerComparison;
