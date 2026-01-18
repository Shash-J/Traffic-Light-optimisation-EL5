import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface NavItem {
    id: string;
    label: string;
    path: string;
    icon: string;
}

const navItems: NavItem[] = [
    { id: 'control', label: 'CONTROL', path: '/dashboard', icon: '⚡' },
    { id: 'analytics', label: 'ANALYTICS', path: '/comparison', icon: '📊' },
    { id: 'scenarios', label: 'SCENARIOS', path: '/scenarios', icon: '🔄' },
    { id: 'logic', label: 'LOGIC_CORE', path: '/agent', icon: '🧠' },
    { id: 'logs', label: 'LOGS', path: '/results', icon: '📋' }
];

interface NavHeaderProps {
    onNewDeployment?: () => void;
}

export const NavHeader: React.FC<NavHeaderProps> = ({ onNewDeployment }) => {
    const location = useLocation();

    const isActive = (path: string) => {
        return location.pathname === path;
    };

    return (
        <header className="nav-header">
            <div className="nav-brand">
                <div className="brand-logo">TO</div>
                <div>
                    <span className="brand-text">
                        TRAFFIC<span className="highlight">OPTIMIZER</span>
                    </span>
                    <span className="brand-version">ADAPTIVE CONTROL SYSTEM V1.0</span>
                </div>
            </div>

            <nav className="nav-tabs">
                {navItems.map(item => (
                    <Link
                        key={item.id}
                        to={item.path}
                        className={`nav-tab ${isActive(item.path) ? 'active' : ''}`}
                    >
                        <span className="icon">{item.icon}</span>
                        {item.label}
                    </Link>
                ))}
            </nav>

            <div className="nav-actions">
                <button
                    className="action-btn primary"
                    onClick={onNewDeployment}
                >
                    ⊕ NEW DEPLOYMENT
                </button>
            </div>
        </header>
    );
};
