import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { NavHeader } from '../components/NavHeader';
import { StatusBar } from '../components/StatusBar';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

interface AgentMetrics {
    episode: number;
    reward: number;
    epsilon: number;
    loss: number;
}

export const AgentInsights: React.FC = () => {
    const navigate = useNavigate();
    const [agentData, setAgentData] = useState<AgentMetrics[]>([]);
    const [currentPolicy, setCurrentPolicy] = useState({
        exploration: 0.15,
        learningRate: 0.001,
        discountFactor: 0.99,
        batchSize: 64,
        memorySize: 10000
    });

    useEffect(() => {
        // Generate sample training data
        const data: AgentMetrics[] = [];
        for (let i = 0; i < 50; i++) {
            data.push({
                episode: i + 1,
                reward: -100 + (i * 3) + Math.random() * 20,
                epsilon: Math.max(0.1, 1 - i * 0.02),
                loss: Math.max(0.01, 1 - i * 0.015 + Math.random() * 0.1)
            });
        }
        setAgentData(data);
    }, []);

    const radarData = [
        { subject: 'Throughput', A: 85, fullMark: 100 },
        { subject: 'Latency', A: 72, fullMark: 100 },
        { subject: 'Stability', A: 90, fullMark: 100 },
        { subject: 'Adaptability', A: 78, fullMark: 100 },
        { subject: 'Efficiency', A: 88, fullMark: 100 },
    ];

    return (
        <div className="dashboard-container">
            <NavHeader onNewDeployment={() => navigate('/junctions')} />

            <div className="analytics-page">
                {/* Header */}
                <div className="analytics-header">
                    <div>
                        <div className="page-label">NEURAL NETWORK DIAGNOSTICS</div>
                        <h1 className="page-title">LOGIC_CORE MONITOR</h1>
                    </div>
                    <div className="actions">
                        <button className="btn-secondary">
                            EXPORT_WEIGHTS
                        </button>
                        <button className="btn-secondary">
                            RESET_AGENT
                        </button>
                    </div>
                </div>

                {/* Main Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', marginBottom: '24px' }}>
                    {/* Training Progress */}
                    <div className="chart-card">
                        <div className="chart-header">
                            <div className="chart-label">TRAINING PROGRESS</div>
                            <div className="chart-title">REWARD CONVERGENCE</div>
                        </div>
                        <div className="chart-content">
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={agentData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                                    <XAxis
                                        dataKey="episode"
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
                                    <Line
                                        type="monotone"
                                        dataKey="reward"
                                        stroke="#00e5ff"
                                        strokeWidth={2}
                                        dot={false}
                                        name="Reward"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Policy Radar */}
                    <div className="chart-card">
                        <div className="chart-header">
                            <div className="chart-label">POLICY EVALUATION</div>
                            <div className="chart-title">PERFORMANCE PROFILE</div>
                        </div>
                        <div className="chart-content">
                            <ResponsiveContainer width="100%" height={250}>
                                <RadarChart data={radarData}>
                                    <PolarGrid stroke="#1f2937" />
                                    <PolarAngleAxis
                                        dataKey="subject"
                                        tick={{ fill: '#9aa0a6', fontSize: 10 }}
                                    />
                                    <PolarRadiusAxis
                                        angle={30}
                                        domain={[0, 100]}
                                        tick={{ fill: '#5f6368', fontSize: 8 }}
                                    />
                                    <Radar
                                        name="Agent"
                                        dataKey="A"
                                        stroke="#00e5ff"
                                        fill="#00e5ff"
                                        fillOpacity={0.3}
                                    />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Hyperparameters */}
                <div className="section-card" style={{ marginBottom: '24px' }}>
                    <div className="section-header">
                        <div>
                            <div className="section-label">CONFIGURATION</div>
                            <div className="section-title">HYPERPARAMETERS</div>
                        </div>
                    </div>
                    <div className="section-content">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px' }}>
                            <HyperparamCard
                                label="EXPLORATION (ε)"
                                value={currentPolicy.exploration.toFixed(2)}
                            />
                            <HyperparamCard
                                label="LEARNING RATE"
                                value={currentPolicy.learningRate.toFixed(4)}
                            />
                            <HyperparamCard
                                label="DISCOUNT (γ)"
                                value={currentPolicy.discountFactor.toFixed(2)}
                            />
                            <HyperparamCard
                                label="BATCH SIZE"
                                value={currentPolicy.batchSize.toString()}
                            />
                            <HyperparamCard
                                label="MEMORY SIZE"
                                value={currentPolicy.memorySize.toLocaleString()}
                            />
                        </div>
                    </div>
                </div>

                {/* Loss & Epsilon Charts */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                    <div className="chart-card">
                        <div className="chart-header">
                            <div className="chart-label">OPTIMIZATION</div>
                            <div className="chart-title">LOSS CURVE</div>
                        </div>
                        <div className="chart-content">
                            <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={agentData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                                    <XAxis
                                        dataKey="episode"
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
                                    <Line
                                        type="monotone"
                                        dataKey="loss"
                                        stroke="#ff6b6b"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="chart-card">
                        <div className="chart-header">
                            <div className="chart-label">EXPLORATION</div>
                            <div className="chart-title">EPSILON DECAY</div>
                        </div>
                        <div className="chart-content">
                            <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={agentData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                                    <XAxis
                                        dataKey="episode"
                                        stroke="#5f6368"
                                        tick={{ fill: '#9aa0a6', fontSize: 10 }}
                                    />
                                    <YAxis
                                        stroke="#5f6368"
                                        tick={{ fill: '#9aa0a6', fontSize: 10 }}
                                        domain={[0, 1]}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#141920',
                                            border: '1px solid #1f2937',
                                            borderRadius: '4px',
                                            fontFamily: 'JetBrains Mono'
                                        }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="epsilon"
                                        stroke="#a855f7"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>

            <StatusBar
                decisionLogic="PPO (VER 3.1)"
                memory="1.2GB"
                load="TRAINING"
            />
        </div>
    );
};

const HyperparamCard = ({ label, value }: { label: string; value: string }) => (
    <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '16px',
        textAlign: 'center'
    }}>
        <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            fontWeight: 700,
            color: 'var(--accent-cyan)',
            marginBottom: '8px'
        }}>
            {value}
        </div>
        <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.6rem',
            color: 'var(--text-muted)',
            letterSpacing: '0.1em',
            textTransform: 'uppercase'
        }}>
            {label}
        </div>
    </div>
);
