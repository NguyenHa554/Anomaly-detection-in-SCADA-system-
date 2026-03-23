import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Upload, Bell } from 'lucide-react';

const STATE_LABELS = {
    connected: 'System Online',
    connecting: 'Connecting…',
    disconnected: 'Disconnected',
    error: 'Connection Error',
    offline: 'Backend Offline',
};

export default function Header({ connectionState = 'connecting', systemStatus = 'normal', activeAlerts = 0 }) {
    const dotClass = `status-dot ${connectionState !== 'connected' ? (connectionState === 'connecting' ? 'connecting' : 'offline')
            : systemStatus === 'attack' ? 'attacking' : ''
        }`;

    const statusLabel = connectionState !== 'connected'
        ? STATE_LABELS[connectionState]
        : systemStatus === 'attack'
            ? `⚠ Attack Detected`
            : 'System Normal';

    return (
        <header className="header">
            <div className="header-inner">
                {/* Brand */}
                <NavLink to="/" className="header-brand" style={{ textDecoration: 'none' }}>
                    <div className="header-logo">S</div>
                    <div>
                        <div className="header-title">SCADA Guard</div>
                        <div className="header-subtitle">Real-time Anomaly Detection</div>
                    </div>
                </NavLink>

                {/* Navigation */}
                <nav className="header-nav">
                    <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                        <LayoutDashboard size={15} />
                        Dashboard
                    </NavLink>
                    <NavLink to="/alerts" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                        <Bell size={15} />
                        Alerts
                        {activeAlerts > 0 && (
                            <span style={{ background: '#ef4444', color: '#fff', borderRadius: '999px', padding: '1px 6px', fontSize: '0.65rem', fontWeight: 700 }}>
                                {activeAlerts}
                            </span>
                        )}
                    </NavLink>
                    <NavLink to="/upload" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                        <Upload size={15} />
                        Upload
                    </NavLink>
                </nav>

                {/* WS Status pill */}
                <div className="header-status">
                    <span className={dotClass} />
                    {statusLabel}
                </div>
            </div>
        </header>
    );
}
