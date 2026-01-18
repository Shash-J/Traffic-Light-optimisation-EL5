import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export const Landing: React.FC = () => {
    const navigate = useNavigate();
    const [terminalLines, setTerminalLines] = useState<string[]>([]);
    const [systemStatus, setSystemStatus] = useState({
        connection: 'SECURE_CONNECTED',
        latency: '32ms',
        region: 'AP-SOUTH-1'
    });

    useEffect(() => {
        // Simulate terminal boot sequence
        const lines = [
            '> Connecting to SUMO backend kernel...',
            '> Verifying heuristic modules...',
            '> System ready. Awaiting user input.'
        ];

        lines.forEach((line, index) => {
            setTimeout(() => {
                setTerminalLines(prev => [...prev, line]);
            }, (index + 1) * 800);
        });
    }, []);

    return (
        <div className="landing-page">
            {/* Grid pattern background */}
            <div className="grid-pattern"></div>

            {/* Main content */}
            <div className="landing-content">
                {/* System Online Badge */}
                <div className="system-badge">
                    <span className="status-dot"></span>
                    SYSTEM ONLINE
                </div>

                {/* Main Title */}
                <h1 className="landing-title">
                    <span className="title-traffic">TRAFFIC</span>{' '}
                    <span className="title-optimizer">OPTIMIZER</span>
                </h1>

                {/* Subtitle */}
                <p className="landing-subtitle">
                    Real-Time Urban Traffic Intelligence & Control System
                </p>

                {/* CTA Button */}
                <button
                    className="cta-button"
                    onClick={() => navigate('/junctions')}
                >
                    ENTER CONTROL CENTER
                    <span className="arrow">→</span>
                </button>
            </div>

            {/* Terminal Console */}
            <div className="terminal-console">
                {terminalLines.map((line, index) => (
                    <div key={index} className="terminal-line" style={{ animationDelay: `${index * 0.5}s` }}>
                        {line.startsWith('> System ready') ? (
                            <>
                                <span className="prefix">&gt;</span>
                                <span className="success">{line.substring(2)}</span>
                            </>
                        ) : (
                            <>
                                <span className="prefix">&gt;</span>
                                <span className="info">{line.substring(2)}</span>
                            </>
                        )}
                    </div>
                ))}
                {terminalLines.length >= 3 && (
                    <div className="terminal-line" style={{ animationDelay: '2.5s' }}>
                        <span className="prefix">&gt;</span>
                        <span className="terminal-cursor"></span>
                    </div>
                )}

                {/* Status indicators */}
                <div className="terminal-status">
                    <div>
                        <span>SECURE CONNECTION: </span>
                        <span className="value">V1.1.3</span>
                    </div>
                    <div>
                        <span>LATENCY: </span>
                        <span className="value">{systemStatus.latency}</span>
                    </div>
                    <div>
                        <span>REGION: </span>
                        <span className="value">{systemStatus.region}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
