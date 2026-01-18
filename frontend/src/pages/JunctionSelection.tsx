import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getLocations, startSimulation, LocationsMap } from '../services/api';

interface JunctionData {
    id: string;
    code: string;
    name: string;
    lanes: number;
    conflictPts: number;
    flowRate: string;
    load: 'critical' | 'high' | 'moderate';
    description: string;
}

const junctionDefaults: Record<string, Partial<JunctionData>> = {
    silk_board: {
        code: 'BLR-SB-01',
        lanes: 48,
        conflictPts: 12,
        flowRate: '12k v/h',
        load: 'critical',
        description: 'Primary arterial intersection. High latency expected.'
    },
    tin_factory: {
        code: 'BLR-TF-02',
        lanes: 36,
        conflictPts: 8,
        flowRate: '10.5k v/h',
        load: 'high',
        description: 'Complex multi-modal convergence zone.'
    },
    hebbal_flyover: {
        code: 'BLR-HF-03',
        lanes: 42,
        conflictPts: 6,
        flowRate: '11k v/h',
        load: 'high',
        description: 'Major flyover junction with high-speed lanes.'
    },
    kr_puram_jct: {
        code: 'BLR-KP-04',
        lanes: 32,
        conflictPts: 14,
        flowRate: '9.5k v/h',
        load: 'moderate',
        description: 'Eastern corridor hub with railway crossing.'
    }
};

export const JunctionSelection: React.FC = () => {
    const navigate = useNavigate();
    const [locations, setLocations] = useState<LocationsMap>({});
    const [selectedLocation, setSelectedLocation] = useState<string>('silk_board');
    const [loading, setLoading] = useState(false);
    const [isConnected, setIsConnected] = useState(false);

    useEffect(() => {
        loadLocations();
    }, []);

    const loadLocations = async () => {
        try {
            const data = await getLocations();
            setLocations(data);
            setIsConnected(true);
            if (Object.keys(data).length > 0) {
                setSelectedLocation(Object.keys(data)[0]);
            }
        } catch (err) {
            console.error("Failed to load locations", err);
            setIsConnected(false);
        }
    };

    const handleStart = async () => {
        setLoading(true);
        try {
            localStorage.setItem('lastLocation', selectedLocation);
            localStorage.setItem('lastIntensity', 'peak');
            await startSimulation('fixed', true, 'peak', selectedLocation);
            navigate('/dashboard');
        } catch (err) {
            console.error("Failed to start", err);
            alert("Failed to start simulation. Check console.");
        } finally {
            setLoading(false);
        }
    };

    const handleJunctionClick = (key: string) => {
        setSelectedLocation(key);
    };

    const getJunctionData = (key: string, loc: any): JunctionData => {
        const defaults = junctionDefaults[key] || junctionDefaults.silk_board;
        return {
            id: key,
            code: defaults.code || 'BLR-XX-00',
            name: loc?.name?.toUpperCase().replace(/ /g, '_') || key.toUpperCase().replace(/ /g, '_'),
            lanes: defaults.lanes || 24,
            conflictPts: defaults.conflictPts || 8,
            flowRate: defaults.flowRate || '8k v/h',
            load: defaults.load || 'moderate',
            description: loc?.description || defaults.description || 'Traffic intersection.'
        };
    };

    return (
        <div className="junction-page">
            {/* Page Header */}
            <div className="page-header">
                <div
                    className="page-breadcrumb"
                    onClick={() => navigate('/')}
                >
                    ← EXIT_SELECTION
                </div>
                <div className="page-label">SYSTEM CONFIGURATION</div>
                <h1 className="page-title">SELECT NETWORK TOPOLOGY</h1>
            </div>

            {/* Junction Grid */}
            <div className="junction-grid">
                {Object.entries(locations).map(([key, loc]) => {
                    const junction = getJunctionData(key, loc);
                    const isSelected = selectedLocation === key;

                    return (
                        <div
                            key={key}
                            className={`junction-card ${isSelected ? 'selected' : ''}`}
                            onClick={() => handleJunctionClick(key)}
                        >
                            <div className="junction-header">
                                <div>
                                    <div className="junction-id">{junction.code}</div>
                                    <div className="junction-name">{junction.name}</div>
                                </div>
                                <div className={`load-badge ${junction.load}`}>
                                    LOAD: {junction.load.toUpperCase()}
                                </div>
                            </div>

                            <div className="junction-stats">
                                <div className="stat-item">
                                    <div className="stat-label">LANES</div>
                                    <div className="stat-value">{junction.lanes}</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">CONFLICT PTS</div>
                                    <div className="stat-value">{junction.conflictPts}</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">FLOW RATE</div>
                                    <div className="stat-value flow">{junction.flowRate}</div>
                                </div>
                            </div>

                            <div className="junction-description">
                                {junction.description}
                            </div>

                            <div className="junction-icon">⬡</div>
                        </div>
                    );
                })}

                {/* Show placeholder cards if no locations loaded */}
                {Object.keys(locations).length === 0 && (
                    <>
                        {Object.entries(junctionDefaults).map(([key, defaults]) => {
                            const junction: JunctionData = {
                                id: key,
                                code: defaults.code || 'BLR-XX-00',
                                name: key.toUpperCase().replace(/_/g, '_'),
                                lanes: defaults.lanes || 24,
                                conflictPts: defaults.conflictPts || 8,
                                flowRate: defaults.flowRate || '8k v/h',
                                load: defaults.load || 'moderate',
                                description: defaults.description || 'Traffic intersection.'
                            };
                            const isSelected = selectedLocation === key;

                            return (
                                <div
                                    key={key}
                                    className={`junction-card ${isSelected ? 'selected' : ''}`}
                                    onClick={() => handleJunctionClick(key)}
                                >
                                    <div className="junction-header">
                                        <div>
                                            <div className="junction-id">{junction.code}</div>
                                            <div className="junction-name">{junction.name}</div>
                                        </div>
                                        <div className={`load-badge ${junction.load}`}>
                                            LOAD: {junction.load.toUpperCase()}
                                        </div>
                                    </div>

                                    <div className="junction-stats">
                                        <div className="stat-item">
                                            <div className="stat-label">LANES</div>
                                            <div className="stat-value">{junction.lanes}</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">CONFLICT PTS</div>
                                            <div className="stat-value">{junction.conflictPts}</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">FLOW RATE</div>
                                            <div className="stat-value flow">{junction.flowRate}</div>
                                        </div>
                                    </div>

                                    <div className="junction-description">
                                        {junction.description}
                                    </div>

                                    <div className="junction-icon">⬡</div>
                                </div>
                            );
                        })}
                    </>
                )}
            </div>

            {/* Control Button */}
            {selectedLocation && (
                <div style={{
                    position: 'fixed',
                    bottom: '60px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 50
                }}>
                    <button
                        className="cta-button"
                        onClick={handleStart}
                        disabled={loading}
                        style={{ opacity: loading ? 0.6 : 1 }}
                    >
                        {loading ? 'INITIALIZING SUMO...' : 'LAUNCH SIMULATION →'}
                    </button>
                </div>
            )}

            {/* Status Bar */}
            <div className="status-bar">
                <div className="status-left">
                    <div className="status-item">
                        <span className={`dot ${isConnected ? 'offline' : 'offline'}`}></span>
                        <span>{isConnected ? 'DISCONNECTED' : 'DISCONNECTED'}</span>
                    </div>
                    <div className="status-item">
                        <span className="label">⚡ ENGINE:</span>
                        <span className="value">READY</span>
                    </div>
                    <div className="status-item">
                        <span className="label">⏱ T-DELTA:</span>
                        <span className="value">0MS</span>
                    </div>
                </div>

                <div className="status-center">
                    <div className="status-item">
                        <span className="label">⚙ DECISION LOGIC:</span>
                        <span className="value">HEURISTIC (VER 2.4)</span>
                    </div>
                </div>

                <div className="status-right">
                    <div className="status-item">
                        <span className="label">💾 MEM:</span>
                        <span className="value">452MB</span>
                    </div>
                    <div className="status-item">
                        <span className="label">LOAD:</span>
                        <span className="value">STANDBY</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
