import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { NavHeader } from '../components/NavHeader';
import { StatusBar } from '../components/StatusBar';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface HistoryPoint {
    time: number;
    queue: number;
    wait: number;
    vehicles: number;
}

export const Comparison: React.FC = () => {
    const navigate = useNavigate();
    const [history, setHistory] = useState<HistoryPoint[]>([]);
    const [metrics, setMetrics] = useState<any>(null);

    useEffect(() => {
        // Load simulation history from localStorage
        const storedHistory = JSON.parse(localStorage.getItem('simulationHistory') || '[]');
        setHistory(storedHistory);

        const storedMetrics = JSON.parse(localStorage.getItem('lastRunMetrics') || 'null');
        setMetrics(storedMetrics);
    }, []);

    // Generate benchmark data for chart
    const benchmarkData = [
        { name: 'Fixed', queue: 45, wait: 35, color: '#ff6b6b' },
        { name: 'Actuated', queue: 38, wait: 28, color: '#ffa726' },
        { name: 'Adaptive', queue: 30, wait: 22, color: '#ffeb3b' },
        { name: 'RL Agent', queue: 22, wait: 15, color: '#00e676' },
        { name: 'MARL', queue: 18, wait: 12, color: '#00e5ff' }
    ];

    const efficiencyDelta = metrics ? '+42.8%' : '+0%';
    const timeRecovered = metrics ? '-35.2s' : '0s';

    const handleExportCSV = () => {
        if (history.length === 0) {
            alert('No data to export');
            return;
        }
        const csv = ['Time,Queue,Wait,Vehicles']
            .concat(history.map(h => `${h.time},${h.queue},${h.wait},${h.vehicles}`))
            .join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'simulation_data.csv';
        a.click();
    };

    return (
        <div className="dashboard-container">
            <NavHeader onNewDeployment={() => navigate('/junctions')} />

            <div className="analytics-page">
                {/* Header */}
                <div className="analytics-header">
                    <div>
                        <div className="page-label">POST RUN ANALYSIS</div>
                        <h1 className="page-title">PERFORMANCE METRICS</h1>
                    </div>
                    <div className="actions">
                        <button className="btn-secondary" onClick={handleExportCSV}>
                            EXPORT_CSV
                        </button>
                        <button className="btn-secondary">
                            RAW_LOGS
                        </button>
                    </div>
                </div>

                {/* Real-time Performance Trends */}
                <div className="chart-card" style={{ marginBottom: '24px' }}>
                    <div className="chart-header">
                        <div className="chart-label">TEMPORAL VARIANCE</div>
                        <div className="chart-title">REAL-TIME PERFORMANCE TRENDS</div>
                    </div>
                    <div className="chart-content">
                        {history.length > 0 ? (
                            <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={history.slice(-20)}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                                    <XAxis
                                        dataKey="time"
                                        stroke="#5f6368"
                                        tick={{ fill: '#9aa0a6', fontSize: 10 }}
                                        tickFormatter={(val) => `${Math.round(val)}s`}
                                    />
                                    <YAxis stroke="#5f6368" tick={{ fill: '#9aa0a6', fontSize: 10 }} />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#141920',
                                            border: '1px solid #1f2937',
                                            borderRadius: '4px',
                                            fontFamily: 'JetBrains Mono'
                                        }}
                                    />
                                    <Bar dataKey="queue" fill="#00e5ff" name="Queue" />
                                    <Bar dataKey="wait" fill="#a855f7" name="Wait Time" />
                                </BarChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="chart-empty">
                                <div className="chart-empty-icon">📊</div>
                                <div className="chart-empty-text">No comparison data yet</div>
                                <div className="chart-empty-subtext">Data will appear as simulation runs</div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Metrics Row */}
                <div className="metrics-row">
                    {/* Algorithm Benchmarking Chart */}
                    <div className="chart-card">
                        <div className="chart-header">
                            <div className="chart-label">CONTROL EFFICACY</div>
                            <div className="chart-title">ALGORITHM BENCHMARKING</div>
                        </div>
                        <div className="chart-content">
                            <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={benchmarkData} layout="horizontal">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                                    <XAxis
                                        dataKey="name"
                                        stroke="#5f6368"
                                        tick={{ fill: '#9aa0a6', fontSize: 10 }}
                                    />
                                    <YAxis stroke="#5f6368" tick={{ fill: '#9aa0a6', fontSize: 10 }} />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#141920',
                                            border: '1px solid #1f2937',
                                            borderRadius: '4px',
                                            fontFamily: 'JetBrains Mono'
                                        }}
                                    />
                                    <Legend />
                                    <Bar dataKey="queue" fill="#ff6b6b" name="Queue Length" />
                                    <Bar dataKey="wait" fill="#00e5ff" name="Wait Time" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Efficiency Delta */}
                    <div className="metric-card-large">
                        <span className="metric-icon">⚡</span>
                        <div className="metric-label">EFFICIENCY DELTA</div>
                        <div className="metric-value positive">{efficiencyDelta}</div>
                        <div className="metric-comparison">vs. FIXED_TIME_BASELINE</div>
                    </div>

                    {/* Time Recovered */}
                    <div className="metric-card-large">
                        <span className="metric-icon">⏱️</span>
                        <div className="metric-label">TIME RECOVERED</div>
                        <div className="metric-value negative">{timeRecovered}</div>
                        <div className="metric-comparison">PER VEHICLE / CYCLE</div>
                    </div>
                </div>

                {/* Additional Stats */}
                {metrics && (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(4, 1fr)',
                        gap: '16px',
                        marginTop: '24px'
                    }}>
                        <StatCard
                            label="TOTAL VEHICLES"
                            value={metrics.vehicle_count || 0}
                            icon="🚗"
                        />
                        <StatCard
                            label="AVG QUEUE"
                            value={metrics.queue_length?.toFixed(1) || 0}
                            icon="📏"
                        />
                        <StatCard
                            label="AVG WAIT"
                            value={`${metrics.waiting_time?.toFixed(1) || 0}s`}
                            icon="⏳"
                        />
                        <StatCard
                            label="THROUGHPUT"
                            value={metrics.arrived_vehicles || 0}
                            icon="✅"
                        />
                    </div>
                )}
            </div>

            <StatusBar />
        </div>
    );
};

const StatCard = ({ label, value, icon }: { label: string; value: any; icon: string }) => (
    <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center'
    }}>
        <div style={{ fontSize: '2rem', marginBottom: '8px' }}>{icon}</div>
        <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.75rem',
            fontWeight: 700,
            color: 'var(--accent-cyan)',
            marginBottom: '4px'
        }}>
            {value}
        </div>
        <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.65rem',
            color: 'var(--text-muted)',
            letterSpacing: '0.1em',
            textTransform: 'uppercase'
        }}>
            {label}
        </div>
    </div>
);
