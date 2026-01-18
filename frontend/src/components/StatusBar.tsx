import React from 'react';

interface StatusBarProps {
    isConnected?: boolean;
    engineStatus?: string;
    tDelta?: string;
    decisionLogic?: string;
    memory?: string;
    load?: string;
}

export const StatusBar: React.FC<StatusBarProps> = ({
    isConnected = false,
    engineStatus = 'READY',
    tDelta = '0MS',
    decisionLogic = 'HEURISTIC (VER 2.4)',
    memory = '452MB',
    load = 'STANDBY'
}) => {
    return (
        <div className="status-bar">
            <div className="status-left">
                <div className="status-item">
                    <span className={`dot ${isConnected ? 'online' : 'offline'}`}></span>
                    <span>{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
                </div>
                <div className="status-item">
                    <span className="label">⚡ ENGINE:</span>
                    <span className="value">{engineStatus}</span>
                </div>
                <div className="status-item">
                    <span className="label">⏱ T-DELTA:</span>
                    <span className="value">{tDelta}</span>
                </div>
            </div>

            <div className="status-center">
                <div className="status-item">
                    <span className="label">⚙ DECISION LOGIC:</span>
                    <span className="value">{decisionLogic}</span>
                </div>
            </div>

            <div className="status-right">
                <div className="status-item">
                    <span className="label">💾 MEM:</span>
                    <span className="value">{memory}</span>
                </div>
                <div className="status-item">
                    <span className="label">LOAD:</span>
                    <span className="value">{load}</span>
                </div>
            </div>
        </div>
    );
};
