import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Upload, Bell } from 'lucide-react';

const STATE_LABELS = {
    connected: 'Hệ thống Trực tuyến',
    connecting: 'Đang kết nối…',
    disconnected: 'Bị ngắt kết nối',
    error: 'Lỗi kết nối',
    offline: 'Backend Ngoại tuyến',
};

export default function Header({ connectionState = 'connecting', systemStatus = 'normal', activeAlerts = 0 }) {
    const dotClass = `status-dot ${connectionState !== 'connected' ? (connectionState === 'connecting' ? 'connecting' : 'offline')
            : systemStatus === 'attack' ? 'attacking' : ''
        }`;

    const statusLabel = connectionState !== 'connected'
        ? STATE_LABELS[connectionState]
        : systemStatus === 'attack'
            ? `⚠ Có sự cố mạng`
            : 'Hệ thống Bình thường';

    return (
        <header className="header">
            <div className="header-inner">
                {/* Brand */}
                <NavLink to="/" className="header-brand" style={{ textDecoration: 'none' }}>
                    <div className="header-logo">S</div>
                    <div>
                        <div className="header-title">SCADA Guard</div>
                        <div className="header-subtitle">Giám sát Bất thường Thời gian thực</div>
                    </div>
                </NavLink>

                {/* Navigation */}
                <nav className="header-nav">
                    <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                        <LayoutDashboard size={15} />
                        Tổng quan
                    </NavLink>
                    <NavLink to="/incidents" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                        <Bell size={15} />
                        Cảnh báo
                        {activeAlerts > 0 && (
                            <span style={{ background: '#ef4444', color: '#fff', borderRadius: '999px', padding: '1px 6px', fontSize: '0.65rem', fontWeight: 700 }}>
                                {activeAlerts}
                            </span>
                        )}
                    </NavLink>
                    <NavLink to="/upload" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                        <Upload size={15} />
                        Tải lên
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
