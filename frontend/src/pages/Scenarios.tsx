import React from 'react';
import { useNavigate } from 'react-router-dom';
import { NavHeader } from '../components/NavHeader';
import { StatusBar } from '../components/StatusBar';

interface Scenario {
    id: string;
    title: string;
    description: string;
    badge: string;
    badgeType: 'warning' | 'danger' | 'info' | 'success';
}

interface ScenarioCategory {
    id: string;
    title: string;
    description: string;
    icon: string;
    iconColor: string;
    scenarios: Scenario[];
}

const scenarioCategories: ScenarioCategory[] = [
    {
        id: 'weather',
        title: 'Weather Impact Analysis',
        description: 'Simulate adverse weather conditions to test system robustness.',
        icon: '🌧️',
        iconColor: 'var(--accent-cyan)',
        scenarios: [
            {
                id: 'heavy_rain',
                title: 'Heavy Rain',
                description: 'Reduced visibility and friction coefficient.',
                badge: '-30% Flow Rate',
                badgeType: 'warning'
            },
            {
                id: 'dense_fog',
                title: 'Dense Fog',
                description: 'Critical safety protocols engaged.',
                badge: '-50% Speed',
                badgeType: 'danger'
            },
            {
                id: 'thunderstorm',
                title: 'Thunderstorm',
                description: 'Severe conditions with potential flooding.',
                badge: '-60% Capacity',
                badgeType: 'danger'
            }
        ]
    },
    {
        id: 'emergency',
        title: 'Emergency Priority',
        description: 'Green corridor generation for high-priority vehicles.',
        icon: '🚨',
        iconColor: 'var(--status-critical)',
        scenarios: [
            {
                id: 'ambulance',
                title: 'Ambulance Override',
                description: 'Forces all conflicting phases to red.',
                badge: 'Immediate Preemption',
                badgeType: 'danger'
            },
            {
                id: 'vip_convoy',
                title: 'VIP Convoy',
                description: 'Synchronized green wave for convoy passage.',
                badge: 'Wave Green',
                badgeType: 'success'
            },
            {
                id: 'fire_truck',
                title: 'Fire Truck Priority',
                description: 'Emergency vehicle signal preemption.',
                badge: 'Code Red',
                badgeType: 'danger'
            }
        ]
    },
    {
        id: 'special',
        title: 'Special Events',
        description: 'Configure for large-scale events and unusual traffic patterns.',
        icon: '🎭',
        iconColor: 'var(--accent-purple)',
        scenarios: [
            {
                id: 'stadium_event',
                title: 'Stadium Evacuation',
                description: 'Mass exodus pattern from venue.',
                badge: '+200% Volume',
                badgeType: 'warning'
            },
            {
                id: 'parade',
                title: 'Street Parade',
                description: 'Road closure with bypass routing.',
                badge: 'Reroute Active',
                badgeType: 'info'
            }
        ]
    },
    {
        id: 'testing',
        title: 'System Testing',
        description: 'Stress test and benchmark scenarios.',
        icon: '🔬',
        iconColor: 'var(--accent-yellow)',
        scenarios: [
            {
                id: 'max_load',
                title: 'Maximum Load Test',
                description: 'Push system to theoretical limits.',
                badge: '100% Stress',
                badgeType: 'danger'
            },
            {
                id: 'sensor_failure',
                title: 'Sensor Failure Simulation',
                description: 'Test fallback control mechanisms.',
                badge: 'Fail-Safe Mode',
                badgeType: 'warning'
            }
        ]
    }
];

export const Scenarios: React.FC = () => {
    const navigate = useNavigate();

    const handleScenarioClick = (categoryId: string, scenarioId: string) => {
        console.log(`Activating scenario: ${categoryId}/${scenarioId}`);
        // TODO: Implement scenario activation
    };

    return (
        <div className="dashboard-container">
            <NavHeader onNewDeployment={() => navigate('/junctions')} />

            <div className="scenarios-page">
                <h1 className="scenarios-title">Emergency & Weather Scenarios</h1>

                <div className="scenarios-grid">
                    {scenarioCategories.map(category => (
                        <div key={category.id} className="scenario-category">
                            <div className="category-header">
                                <span className="category-icon" style={{ color: category.iconColor }}>
                                    {category.icon}
                                </span>
                                <h2 className="category-title">{category.title}</h2>
                            </div>
                            <p className="category-description">{category.description}</p>

                            <div className="scenario-list">
                                {category.scenarios.map(scenario => (
                                    <div
                                        key={scenario.id}
                                        className="scenario-item"
                                        onClick={() => handleScenarioClick(category.id, scenario.id)}
                                    >
                                        <div className="scenario-info">
                                            <h4>{scenario.title}</h4>
                                            <p>{scenario.description}</p>
                                        </div>
                                        <span className={`scenario-badge ${scenario.badgeType}`}>
                                            {scenario.badge}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <StatusBar />
        </div>
    );
};
