import { NavLink } from 'react-router-dom';
import { STAGES, STAGE_CONFIG } from '../constants/stages';

export default function Sidebar() {
    return (
        <aside className="sidebar">
            <NavLink to="/" className="sidebar-brand">
                <div className="sidebar-logo">
                    <span className="material-symbols-outlined">shield</span>
                </div>
                <div>
                    <div className="sidebar-title">SCADA-SECURITY</div>
                    <div className="sidebar-subtitle">Hệ thống quản lý</div>
                </div>
            </NavLink>

            <nav className="sidebar-nav">
                <NavLink to="/" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
                    <span className="material-symbols-outlined">dashboard</span>
                    Tổng quan
                </NavLink>

                <div className="sidebar-group">
                    <div className="sidebar-link" style={{ cursor: 'default', color: 'var(--text-primary)', fontWeight: 600 }}>
                        <span className="material-symbols-outlined">account_tree</span>
                        Giai đoạn
                    </div>
                    <div style={{ paddingLeft: 32, display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
                        {STAGES.map((stage) => (
                            <NavLink
                                key={stage}
                                to={`/stages/${stage}`}
                                className={({ isActive }) => `sidebar-sublink${isActive ? ' active' : ''}`}
                                style={{
                                    padding: '8px 12px',
                                    fontSize: '0.85rem',
                                    color: 'var(--text-secondary)',
                                    textDecoration: 'none',
                                    borderRadius: 6,
                                }}
                            >
                                {stage} - {STAGE_CONFIG[stage].name}
                            </NavLink>
                        ))}
                    </div>
                </div>

                <NavLink to="/incidents" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
                    <span className="material-symbols-outlined">history</span>
                    Nhật ký sự cố
                </NavLink>
                <NavLink to="/reports" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
                    <span className="material-symbols-outlined">bar_chart</span>
                    Báo cáo
                </NavLink>
                <NavLink to="/settings" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
                    <span className="material-symbols-outlined">settings</span>
                    Cài đặt
                </NavLink>
            </nav>

            <div className="sidebar-footer">
                <div className="user-avatar">AD</div>
                <div className="user-info">
                    <div className="user-name">Admin User</div>
                    <div className="user-role">admin@scada.local</div>
                </div>
                <span className="material-symbols-outlined sidebar-logout">logout</span>
            </div>
        </aside>
    );
}
