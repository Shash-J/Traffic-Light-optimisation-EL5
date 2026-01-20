import React, { useState, useEffect } from 'react';
import { Play, Square, Activity, Map, BarChart3, Clock, Zap, Target } from 'lucide-react';
import { Line, Radar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    RadialLinearScale,
    Filler,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    RadialLinearScale,
    Filler,
    Title,
    Tooltip,
    Legend
);

const App = () => {
    const [selectedMap, setSelectedMap] = useState('silkboard');
    const [isRunning, setIsRunning] = useState(false);
    const [stats, setStats] = useState({
        running_time: 0,
        mode: 'single',
        rl: { waiting_time: 0, queue_length: 0, throughput: 0, efficiency: 0 },
        baseline: { waiting_time: 0, queue_length: 0, throughput: 0, efficiency: 0 }
    });

    // History for charts
    const [history, setHistory] = useState({
        labels: [],
        rlWait: [],
        baseWait: []
    });

    const maps = [
        { id: 'silkboard', name: 'Silkboard Junction', desc: 'High Traffic Density' },
        { id: 'hosmat', name: 'Hosmat Junction', desc: 'Dual Simulation (RL vs Baseline)' },
        { id: 'map3', name: 'Indiranagar', desc: 'Commercial Zone', disabled: true },
        { id: 'map4', name: 'MG Road', desc: 'Central Business Dist.', disabled: true },
    ];

    const toggleSimulation = async () => {
        if (isRunning) {
            await fetch('/api/stop-simulation', { method: 'POST' });
            setIsRunning(false);
        } else {
            const res = await fetch('/api/start-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mapId: selectedMap })
            });
            if (res.ok) setIsRunning(true);
        }
    };

    useEffect(() => {
        let interval;
        if (isRunning) {
            interval = setInterval(async () => {
                const res = await fetch('/api/stats');
                const data = await res.json();
                setStats(data);

                // Update History
                if (data.mode === 'dual') {
                    setHistory(prev => {
                        const newLabels = [...prev.labels, data.running_time + 's'];
                        const newRl = [...prev.rlWait, data.rl.waiting_time];
                        const newBase = [...prev.baseWait, data.baseline.waiting_time];

                        // Keep last 30 points
                        if (newLabels.length > 30) {
                            newLabels.shift(); newRl.shift(); newBase.shift();
                        }

                        return { labels: newLabels, rlWait: newRl, baseWait: newBase };
                    });
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isRunning]);

    // --- Components ---

    const MetricCard = ({ title, icon: Icon, value1, label1, value2, label2, unit }) => {
        // Calculate diff logic if both exist
        let improvement = 0;
        if (value2 !== 0) improvement = ((value2 - value1) / value2) * 100; // Assuming value1 is "ours" (better)

        // Invert logic for "Wait Time" and "Queue" where Lower is Better
        const isLowerBetter = title.includes('WAIT') || title.includes('QUEUE');
        const isBetter = isLowerBetter ? value1 < value2 : value1 > value2;
        const colorClass = isBetter ? 'trend-positive' : 'trend-negative';

        // Recalculate percent for display
        const percent = Math.abs(Math.round(((value1 - value2) / value2) * 100));

        return (
            <div className="metric-card">
                <h3><Icon size={16} /> {title}</h3>
                <div className="metric-row">
                    <div>
                        <span className="metric-value">{value1}</span>
                        <span className="metric-sub"> {unit}</span>
                        <div className="metric-sub" style={{ color: '#10b981' }}>RL AGENT</div>
                    </div>
                    <div className="text-right">
                        <span className="metric-value" style={{ color: '#64748b', fontSize: '1.4rem' }}>{value2}</span>
                        <span className="metric-sub"> {unit}</span>
                        <div className="metric-sub">FIXED</div>
                    </div>
                </div>
                <div className={`improvement-badge ${colorClass}`}>
                    {isBetter ? '▲' : '▼'} {percent}% IMPROVEMENT
                </div>
            </div>
        );
    };

    if (isRunning && stats.mode === 'dual') {
        return (
            <div className="app-container" style={{ display: 'block', padding: 0 }}>
                {/* Full Screen Stats Dashboard */}
                <header className="app-header">
                    <div className="brand">
                        <Activity className="text-blue-500" />
                        <h1 style={{ fontSize: '1.2rem', margin: 0 }}>TRAFFIC<span style={{ fontWeight: 300 }}>AI</span></h1>
                    </div>

                    <div className="flex gap-4">
                        <button className="primary-btn" style={{ padding: '0.5rem 1.5rem', width: 'auto' }}>Deep Analytics</button>
                    </div>

                    <div className="flex items-center gap-4">
                        <button className="primary-btn stop" onClick={toggleSimulation} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
                            Stop
                        </button>
                        <div className="time-badge">
                            ● RUNNING {stats.running_time.toFixed(1)}s
                        </div>
                    </div>
                </header>

                <main style={{ padding: '2rem', maxWidth: '1600px', margin: '0 auto' }}>
                    <div className="dashboard-grid">
                        <MetricCard
                            title="AVG WAIT TIME"
                            icon={Clock}
                            value1={stats.rl.waiting_time}
                            value2={stats.baseline.waiting_time}
                            unit="s"
                        />
                        <MetricCard
                            title="THROUGHPUT"
                            icon={Zap}
                            value1={stats.rl.throughput}
                            value2={stats.baseline.throughput}
                            unit="v/m"
                        />
                        <MetricCard
                            title="QUEUE LENGTH"
                            icon={BarChart3}
                            value1={stats.rl.queue_length}
                            value2={stats.baseline.queue_length}
                            unit="veh"
                        />
                        <MetricCard
                            title="EFFICIENCY"
                            icon={Target}
                            value1={stats.rl.efficiency}
                            value2={stats.baseline.efficiency}
                            unit="%"
                        />
                    </div>

                    <div className="charts-row">
                        <div className="chart-panel">
                            <h3><Activity size={16} /> WAIT TIME TRENDS</h3>
                            <div style={{ height: '300px', width: '100%' }}>
                                <Line
                                    data={{
                                        labels: history.labels,
                                        datasets: [
                                            {
                                                label: 'RL Agent',
                                                data: history.rlWait,
                                                borderColor: '#10b981',
                                                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                                                fill: true,
                                                tension: 0.4
                                            },
                                            {
                                                label: 'Fixed Time',
                                                data: history.baseWait,
                                                borderColor: '#ef4444',
                                                backgroundColor: 'rgba(239, 68, 68, 0.05)',
                                                fill: true,
                                                tension: 0.4
                                            }
                                        ]
                                    }}
                                    options={{
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        interaction: { mode: 'index', intersect: false },
                                        scales: {
                                            y: { grid: { color: '#334155' }, beginAtZero: true },
                                            x: { display: false }
                                        },
                                        plugins: { legend: { display: true, position: 'bottom' } }
                                    }}
                                />
                            </div>
                        </div>

                        <div className="chart-panel">
                            <h3><Target size={16} /> PERFORMANCE RADAR</h3>
                            <div style={{ height: '300px', display: 'flex', justifyContent: 'center' }}>
                                <Radar
                                    data={{
                                        labels: ['Efficiency', 'Throughput', 'Queue Safety', 'Wait Time'],
                                        datasets: [
                                            {
                                                label: 'RL Agent',
                                                data: [stats.rl.efficiency, stats.rl.throughput, 100 - stats.rl.queue_length, 100 - stats.rl.waiting_time],
                                                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                                                borderColor: '#10b981',
                                                borderWidth: 2,
                                            },
                                            {
                                                label: 'Fixed',
                                                data: [stats.baseline.efficiency, stats.baseline.throughput, 100 - stats.baseline.queue_length, 100 - stats.baseline.waiting_time],
                                                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                                                borderColor: '#ef4444',
                                                borderWidth: 2,
                                            }
                                        ]
                                    }}
                                    options={{
                                        scales: {
                                            r: {
                                                angleLines: { color: '#334155' },
                                                grid: { color: '#334155' },
                                                pointLabels: { color: '#94a3b8' },
                                                ticks: { display: false }
                                            }
                                        },
                                        plugins: { legend: { position: 'bottom' } }
                                    }}
                                />
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        );
    }

    // --- Standard Selection View ---

    return (
        <div className="app-container">
            <aside className="sidebar">
                <div className="brand">
                    <Activity className="text-blue-500" />
                    <h1>TrafficAI</h1>
                </div>

                <div>
                    <h2 className="text-gray-400 text-sm mb-4 font-semibold tracking-wider">SELECT REGION</h2>
                    <div className="map-grid">
                        {maps.map(map => (
                            <div
                                key={map.id}
                                className={`map-card ${selectedMap === map.id ? 'active' : ''} ${map.disabled ? 'disabled' : ''}`}
                                onClick={() => !map.disabled && setSelectedMap(map.id)}
                            >
                                <div className="flex justify-between items-start">
                                    <h3>{map.name}</h3>
                                    {selectedMap === map.id && <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />}
                                </div>
                                <p>{map.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <div>
                    <h2 className="text-gray-400 text-sm mb-4 font-semibold tracking-wider">CONTROL</h2>
                    <button
                        className={`primary-btn ${isRunning ? 'stop' : ''}`}
                        onClick={toggleSimulation}
                    >
                        {isRunning ? <><Square size={20} /> Stop Simulation</> : <><Play size={20} /> Start Simulation</>}
                    </button>
                </div>
            </aside>

            <main className="main-view">
                <div className="text-center opacity-50">
                    <Map size={64} className="mx-auto mb-4 text-slate-600" />
                    <h2 className="text-2xl font-bold text-slate-600">Initial State</h2>
                    <p>Select a map to begin deep analysis</p>
                </div>
            </main>
        </div>
    );
}

export default App;
