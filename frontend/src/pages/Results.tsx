import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { NavHeader } from '../components/NavHeader';
import { StatusBar } from '../components/StatusBar';

interface LogEntry {
    timestamp: string;
    level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG' | 'SUCCESS';
    module: string;
    message: string;
}

export const Results: React.FC = () => {
    const navigate = useNavigate();
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [filter, setFilter] = useState<string>('ALL');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        // Generate sample logs
        const sampleLogs: LogEntry[] = [
            { timestamp: '10:45:23.456', level: 'INFO', module: 'SUMO_CORE', message: 'Simulation initialized successfully' },
            { timestamp: '10:45:23.789', level: 'SUCCESS', module: 'RL_AGENT', message: 'Model loaded from checkpoint: model_v2.pth' },
            { timestamp: '10:45:24.012', level: 'INFO', module: 'TRAFFIC_GEN', message: 'Vehicle spawning rate set to 0.8/s' },
            { timestamp: '10:45:25.234', level: 'DEBUG', module: 'SIGNAL_CTRL', message: 'Phase transition: NS_GREEN -> EW_GREEN' },
            { timestamp: '10:45:26.567', level: 'WARN', module: 'QUEUE_MON', message: 'Queue length exceeding threshold: 45 vehicles' },
            { timestamp: '10:45:27.890', level: 'INFO', module: 'METRICS', message: 'Throughput: 156 veh/min | Avg Wait: 23.4s' },
            { timestamp: '10:45:28.123', level: 'DEBUG', module: 'RL_AGENT', message: 'Action selected: EXTEND_NS, Q-value: 0.847' },
            { timestamp: '10:45:29.456', level: 'ERROR', module: 'SENSOR', message: 'Loop detector LD_01 communication timeout' },
            { timestamp: '10:45:30.789', level: 'INFO', module: 'FAILSAFE', message: 'Switching to actuated control fallback' },
            { timestamp: '10:45:31.012', level: 'SUCCESS', module: 'SENSOR', message: 'Loop detector LD_01 reconnected' },
            { timestamp: '10:45:32.345', level: 'INFO', module: 'RL_AGENT', message: 'Episode reward: +45.67' },
            { timestamp: '10:45:33.678', level: 'DEBUG', module: 'WEATHER', message: 'Weather condition updated: LIGHT_RAIN' },
            { timestamp: '10:45:34.901', level: 'WARN', module: 'SAFETY', message: 'Emergency vehicle approaching - ETA 45s' },
            { timestamp: '10:45:35.234', level: 'SUCCESS', module: 'PREEMPT', message: 'Green corridor established for emergency vehicle' },
            { timestamp: '10:45:36.567', level: 'INFO', module: 'METRICS', message: 'Simulation time: 300s | Vehicles processed: 892' }
        ];
        setLogs(sampleLogs);
    }, []);

    const getLevelColor = (level: string) => {
        switch (level) {
            case 'ERROR': return 'var(--status-critical)';
            case 'WARN': return 'var(--accent-yellow)';
            case 'SUCCESS': return 'var(--accent-green)';
            case 'DEBUG': return 'var(--accent-purple)';
            default: return 'var(--accent-cyan)';
        }
    };

    const filteredLogs = logs.filter(log => {
        const matchesFilter = filter === 'ALL' || log.level === filter;
        const matchesSearch = searchQuery === '' ||
            log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
            log.module.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesFilter && matchesSearch;
    });

    const handleClearLogs = () => {
        setLogs([]);
    };

    const handleExportLogs = () => {
        const logText = logs.map(l => `[${l.timestamp}] [${l.level}] [${l.module}] ${l.message}`).join('\n');
        const blob = new Blob([logText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'simulation_logs.txt';
        a.click();
    };

    return (
        <div className="dashboard-container">
            <NavHeader onNewDeployment={() => navigate('/junctions')} />

            <div className="analytics-page">
                {/* Header */}
                <div className="analytics-header">
                    <div>
                        <div className="page-label">SYSTEM OUTPUT</div>
                        <h1 className="page-title">EXECUTION LOGS</h1>
                    </div>
                    <div className="actions">
                        <button className="btn-secondary" onClick={handleExportLogs}>
                            EXPORT_LOGS
                        </button>
                        <button className="btn-secondary" onClick={handleClearLogs}>
                            CLEAR_ALL
                        </button>
                    </div>
                </div>

                {/* Filters */}
                <div style={{
                    display: 'flex',
                    gap: '12px',
                    marginBottom: '24px',
                    alignItems: 'center'
                }}>
                    {/* Search */}
                    <div style={{ flex: 1, position: 'relative' }}>
                        <input
                            type="text"
                            placeholder="Search logs..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            style={{
                                width: '100%',
                                padding: '12px 16px',
                                paddingLeft: '40px',
                                background: 'var(--bg-card)',
                                border: '1px solid var(--border-color)',
                                borderRadius: '6px',
                                color: 'var(--text-primary)',
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.85rem',
                                outline: 'none'
                            }}
                        />
                        <span style={{
                            position: 'absolute',
                            left: '14px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            color: 'var(--text-muted)'
                        }}>🔍</span>
                    </div>

                    {/* Level Filter */}
                    {['ALL', 'INFO', 'SUCCESS', 'WARN', 'ERROR', 'DEBUG'].map(level => (
                        <button
                            key={level}
                            onClick={() => setFilter(level)}
                            style={{
                                padding: '10px 16px',
                                background: filter === level ? 'rgba(0, 229, 255, 0.15)' : 'var(--bg-card)',
                                border: filter === level ? '1px solid var(--accent-cyan)' : '1px solid var(--border-color)',
                                borderRadius: '4px',
                                color: filter === level ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.7rem',
                                letterSpacing: '0.1em',
                                cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                        >
                            {level}
                        </button>
                    ))}
                </div>

                {/* Log Output */}
                <div className="section-card">
                    <div className="section-header">
                        <div className="section-label">LOG OUTPUT</div>
                        <div style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.7rem',
                            color: 'var(--text-muted)'
                        }}>
                            {filteredLogs.length} entries
                        </div>
                    </div>
                    <div style={{
                        background: 'var(--bg-primary)',
                        padding: '16px',
                        maxHeight: '500px',
                        overflowY: 'auto',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.8rem',
                        lineHeight: '1.8'
                    }}>
                        {filteredLogs.length > 0 ? (
                            filteredLogs.map((log, index) => (
                                <div
                                    key={index}
                                    style={{
                                        display: 'flex',
                                        gap: '12px',
                                        padding: '8px 0',
                                        borderBottom: '1px solid var(--border-color)'
                                    }}
                                >
                                    <span style={{ color: 'var(--text-muted)', minWidth: '100px' }}>
                                        {log.timestamp}
                                    </span>
                                    <span style={{
                                        color: getLevelColor(log.level),
                                        minWidth: '60px',
                                        fontWeight: 600
                                    }}>
                                        [{log.level}]
                                    </span>
                                    <span style={{
                                        color: 'var(--accent-purple)',
                                        minWidth: '100px'
                                    }}>
                                        [{log.module}]
                                    </span>
                                    <span style={{ color: 'var(--text-primary)' }}>
                                        {log.message}
                                    </span>
                                </div>
                            ))
                        ) : (
                            <div style={{
                                textAlign: 'center',
                                padding: '40px',
                                color: 'var(--text-muted)'
                            }}>
                                No logs to display
                            </div>
                        )}
                    </div>
                </div>

                {/* Stats Summary */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(5, 1fr)',
                    gap: '16px',
                    marginTop: '24px'
                }}>
                    <LogStatCard
                        label="INFO"
                        count={logs.filter(l => l.level === 'INFO').length}
                        color="var(--accent-cyan)"
                    />
                    <LogStatCard
                        label="SUCCESS"
                        count={logs.filter(l => l.level === 'SUCCESS').length}
                        color="var(--accent-green)"
                    />
                    <LogStatCard
                        label="WARN"
                        count={logs.filter(l => l.level === 'WARN').length}
                        color="var(--accent-yellow)"
                    />
                    <LogStatCard
                        label="ERROR"
                        count={logs.filter(l => l.level === 'ERROR').length}
                        color="var(--status-critical)"
                    />
                    <LogStatCard
                        label="DEBUG"
                        count={logs.filter(l => l.level === 'DEBUG').length}
                        color="var(--accent-purple)"
                    />
                </div>
            </div>

            <StatusBar />
        </div>
    );
};

const LogStatCard = ({ label, count, color }: { label: string; count: number; color: string }) => (
    <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '16px',
        textAlign: 'center',
        borderTop: `3px solid ${color}`
    }}>
        <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '2rem',
            fontWeight: 700,
            color: color,
            marginBottom: '4px'
        }}>
            {count}
        </div>
        <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.65rem',
            color: 'var(--text-muted)',
            letterSpacing: '0.1em'
        }}>
            {label}
        </div>
    </div>
);
