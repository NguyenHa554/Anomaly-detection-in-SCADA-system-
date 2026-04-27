import { useState } from 'react';
import { STAGES, STAGE_CONFIG, MONITORING_ONLY_STAGES } from '../constants/stages';

const MOCK_USERS = [
    { id: 1, name: 'Nguyen Van A', role: 'Admin', status: 'Active', initials: 'NV' },
    { id: 2, name: 'Tran Thi B', role: 'Ky thuat vien', status: 'Active', initials: 'TB' },
    { id: 3, name: 'Le Van C', role: 'Giam sat', status: 'Inactive', initials: 'LC' },
];

export default function SettingsPage() {
    const [emailAlerts, setEmailAlerts] = useState(true);
    const [soundAlerts, setSoundAlerts] = useState(false);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <header className="page-header">
                <h1 className="page-title">Cai dat he thong</h1>
                <div className="header-actions">
                    <div className="search-bar">
                        <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: 18 }}>search</span>
                        <input type="text" placeholder="Tim kiem cai dat..." />
                    </div>
                    <button className="header-icon-btn">
                        <span className="material-symbols-outlined">notifications</span>
                    </button>
                </div>
            </header>

            <div className="page-container" style={{ padding: 32, maxWidth: 960, margin: '0 auto' }}>
                <section className="settings-section">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                        <div>
                            <h2 className="settings-section-title">Quan ly tai khoan</h2>
                            <p className="settings-section-desc" style={{ marginBottom: 0 }}>
                                Quan ly nguoi dung truy cap va phan quyen he thong
                            </p>
                        </div>
                        <button className="btn btn-primary">
                            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>add</span>
                            Them nguoi dung
                        </button>
                    </div>

                    <div className="data-table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Ten nguoi dung</th>
                                    <th>Vai tro</th>
                                    <th>Trang thai</th>
                                    <th>Thao tac</th>
                                </tr>
                            </thead>
                            <tbody>
                                {MOCK_USERS.map((user) => (
                                    <tr key={user.id}>
                                        <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                                            <span
                                                className="avatar-small"
                                                style={{
                                                    background: user.status === 'Active' ? '#e0f2fe' : 'var(--bg-base)',
                                                    color: user.status === 'Active' ? '#0369a1' : 'var(--text-muted)',
                                                }}
                                            >
                                                {user.initials}
                                            </span>
                                            {user.name}
                                        </td>
                                        <td>
                                            <span
                                                style={{
                                                    fontSize: '0.75rem',
                                                    padding: '2px 8px',
                                                    borderRadius: 999,
                                                    background: 'var(--bg-base)',
                                                    border: '1px solid var(--border-subtle)',
                                                }}
                                            >
                                                {user.role}
                                            </span>
                                        </td>
                                        <td>
                                            {user.status === 'Active' ? (
                                                <span className="status-badge success">
                                                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }}></span>
                                                    Active
                                                </span>
                                            ) : (
                                                <span className="status-badge" style={{ color: 'var(--text-muted)' }}>
                                                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }}></span>
                                                    Inactive
                                                </span>
                                            )}
                                        </td>
                                        <td>
                                            <button className="btn-ghost" style={{ fontSize: '0.8rem', padding: 0 }}>Chinh sua</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                <section className="settings-section">
                    <h2 className="settings-section-title">Cau hinh canh bao</h2>
                    <p className="settings-section-desc">Thiet lap cach nhan thong bao khi co su co</p>

                    <div className="toggle-row">
                        <div>
                            <div className="toggle-label">Gui email khi co canh bao CRITICAL</div>
                            <div className="toggle-desc">
                                He thong se tu dong gui thong bao den cac dia chi email da dang ky
                            </div>
                        </div>
                        <label className="switch">
                            <input type="checkbox" checked={emailAlerts} onChange={(e) => setEmailAlerts(e.target.checked)} />
                            <span className="slider"></span>
                        </label>
                    </div>

                    <div className="toggle-row">
                        <div>
                            <div className="toggle-label">Bat am thanh bao dong</div>
                            <div className="toggle-desc">
                                Phat tin hieu am thanh truc tiep tren bang dieu khien giam sat
                            </div>
                        </div>
                        <label className="switch">
                            <input type="checkbox" checked={soundAlerts} onChange={(e) => setSoundAlerts(e.target.checked)} />
                            <span className="slider"></span>
                        </label>
                    </div>
                </section>

                <section className="settings-section">
                    <h2 className="settings-section-title">Thong tin tram P1-P6</h2>
                    <p className="settings-section-desc">
                        Trang thai ket noi thoi gian thuc cua cac tram dieu khien. {MONITORING_ONLY_STAGES.join(', ')} chi hien thi dashboard va phan tich, khong kich hoat canh bao cuoi.
                    </p>

                    <div className="station-grid">
                        {STAGES.map((stage) => (
                            <div className="station-card" key={stage}>
                                <span className="material-symbols-outlined">cell_tower</span>
                                <div>
                                    <div className="station-title">TRAM {stage}</div>
                                    <div className="station-name">{STAGE_CONFIG[stage].name}</div>
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                                    <div className="station-status-badge">DA KET NOI</div>
                                    {!STAGE_CONFIG[stage].monitored && (
                                        <div style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.04em', color: 'var(--text-muted)' }}>
                                            MONITORING ONLY
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                <div style={{ textAlign: 'center', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 48 }}>
                    SCADA-SECURITY 2026 | v4.2.0-LTS
                </div>
            </div>
        </div>
    );
}
