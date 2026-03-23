import { useState } from 'react';

const MOCK_USERS = [
    { id: 1, name: 'Nguyễn Văn A', role: 'Admin', status: 'Active', initials: 'NV' },
    { id: 2, name: 'Trần Thị B', role: 'Kỹ thuật viên', status: 'Active', initials: 'TB' },
    { id: 3, name: 'Lê Văn C', role: 'Giám sát', status: 'Inactive', initials: 'LC' },
];

const STAGES = [
    { id: 'P1', name: 'Hệ thống Lọc thô' },
    { id: 'P2', name: 'Bể phản ứng hóa chất' },
    { id: 'P3', name: 'Hệ thống Bơm cao áp' },
    { id: 'P4', name: 'Module Xử lý vi sinh' },
    { id: 'P5', name: 'Kiểm định chất lượng' },
    { id: 'P6', name: 'Điều phối đầu ra' },
];

export default function SettingsPage() {
    const [emailAlerts, setEmailAlerts] = useState(true);
    const [soundAlerts, setSoundAlerts] = useState(false);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <header className="page-header">
                <h1 className="page-title">Cài đặt Hệ thống</h1>
                <div className="header-actions">
                    <div className="search-bar">
                        <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: 18 }}>search</span>
                        <input type="text" placeholder="Tìm kiếm cài đặt..." />
                    </div>
                    <button className="header-icon-btn">
                        <span className="material-symbols-outlined">notifications</span>
                    </button>
                </div>
            </header>

            <div className="page-container" style={{ padding: '32px', maxWidth: 960, margin: '0 auto' }}>

                {/* ── Quản lý Tài khoản ─────────────────────────────────────────────── */}
                <section className="settings-section">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                        <div>
                            <h2 className="settings-section-title">Quản lý Tài khoản</h2>
                            <p className="settings-section-desc" style={{ marginBottom: 0 }}>Quản lý người dùng truy cập và phân quyền hệ thống</p>
                        </div>
                        <button className="btn btn-primary">
                            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>add</span>
                            Thêm người dùng
                        </button>
                    </div>

                    <div className="data-table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Tên người dùng</th>
                                    <th>Vai trò</th>
                                    <th>Trạng thái</th>
                                    <th>Thao tác</th>
                                </tr>
                            </thead>
                            <tbody>
                                {MOCK_USERS.map(u => (
                                    <tr key={u.id}>
                                        <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                                            <span className="avatar-small" style={{ background: u.status === 'Active' ? '#e0f2fe' : 'var(--bg-base)', color: u.status === 'Active' ? '#0369a1' : 'var(--text-muted)' }}>{u.initials}</span>
                                            {u.name}
                                        </td>
                                        <td>
                                            <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: 999, background: 'var(--bg-base)', border: '1px solid var(--border-subtle)' }}>
                                                {u.role}
                                            </span>
                                        </td>
                                        <td>
                                            {u.status === 'Active' ? (
                                                <span className="status-badge success"><span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }}></span> Active</span>
                                            ) : (
                                                <span className="status-badge" style={{ color: 'var(--text-muted)' }}><span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }}></span> Inactive</span>
                                            )}
                                        </td>
                                        <td>
                                            <button className="btn-ghost" style={{ fontSize: '0.8rem', padding: 0 }}>Chỉnh sửa</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* ── Cấu hình Cảnh báo ─────────────────────────────────────────────── */}
                <section className="settings-section">
                    <h2 className="settings-section-title">Cấu hình Cảnh báo</h2>
                    <p className="settings-section-desc">Thiết lập cách thức nhận thông báo khi có sự cố</p>

                    <div className="toggle-row">
                        <div>
                            <div className="toggle-label">Gửi email khi có cảnh báo CRITICAL</div>
                            <div className="toggle-desc">Hệ thống sẽ tự động gửi thông báo đến các địa chỉ email đã đăng ký</div>
                        </div>
                        <label className="switch">
                            <input type="checkbox" checked={emailAlerts} onChange={e => setEmailAlerts(e.target.checked)} />
                            <span className="slider"></span>
                        </label>
                    </div>

                    <div className="toggle-row">
                        <div>
                            <div className="toggle-label">Bật âm thanh báo động</div>
                            <div className="toggle-desc">Phát tín hiệu âm thanh trực tiếp trên bảng điều khiển giám sát</div>
                        </div>
                        <label className="switch">
                            <input type="checkbox" checked={soundAlerts} onChange={e => setSoundAlerts(e.target.checked)} />
                            <span className="slider"></span>
                        </label>
                    </div>
                </section>

                {/* ── Thông tin Trạm P1-P6 ─────────────────────────────────────────────── */}
                <section className="settings-section">
                    <h2 className="settings-section-title">Thông tin Trạm P1-P6</h2>
                    <p className="settings-section-desc">Trạng thái kết nối thời gian thực của các trạm điều khiển</p>

                    <div className="station-grid">
                        {STAGES.map(s => (
                            <div className="station-card" key={s.id}>
                                <span className="material-symbols-outlined">cell_tower</span>
                                <div>
                                    <div className="station-title">TRẠM {s.id}</div>
                                    <div className="station-name">{s.name}</div>
                                </div>
                                <div className="station-status-badge">ĐÃ KẾT NỐI</div>
                            </div>
                        ))}
                    </div>
                </section>

                <div style={{ textAlign: 'center', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 48 }}>
                    © 2026 SCADA-SECURITY Hệ thống Giám sát & Quản lý Hạ tầng Nước • v4.2.0-LTS
                </div>
            </div>
        </div>
    );
}
