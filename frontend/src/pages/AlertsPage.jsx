import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, Download } from 'lucide-react';
import { getAlerts, acknowledgeAlert } from '../services/api';

const SEVERITIES = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
const STAGES = ['ALL', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'];

function formatDateTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return (
        <span>
            <strong>{d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</strong><br />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {d.toLocaleDateString('vi-VN')}
            </span>
        </span>
    );
}

export default function AlertsPage() {
    const [alerts, setAlerts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filterSev, setFilterSev] = useState('ALL');
    const [filterStage, setFilterStage] = useState('ALL');

    const fetchAlerts = useCallback(async () => {
        setLoading(true);
        try {
            const params = { limit: 200 };
            const data = await getAlerts(params);
            setAlerts(data.alerts || data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchAlerts(); }, [fetchAlerts]);

    async function handleAck(id) {
        try {
            await acknowledgeAlert(id);
            setAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true } : a));
        } catch (err) {
            console.error(err);
        }
    }

    const filtered = alerts.filter(a => {
        if (filterSev !== 'ALL' && a.severity !== filterSev) return false;
        if (filterStage !== 'ALL' && a.stage !== filterStage) return false;
        return true;
    });

    const unacked = alerts.filter(a => !a.acknowledged).length;
    const resolvedCount = alerts.length - unacked;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Page Header */}
            <header className="page-header">
                <h1 className="page-title">Nhật ký sự cố</h1>
                <div className="header-actions">
                    <div className="search-bar">
                        <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: 18 }}>search</span>
                        <input type="text" placeholder="Tìm kiếm mã sự cố, thiết bị..." />
                    </div>
                    <button className="btn btn-secondary" style={{ padding: '6px 16px', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)' }} onClick={fetchAlerts} disabled={loading}>
                        <RefreshCw size={14} className={loading ? 'spin' : ''} style={{ marginRight: 6 }} />
                        Hôm nay
                    </button>
                    <button className="btn btn-primary" style={{ padding: '6px 16px' }}>
                        <Download size={14} style={{ marginRight: 6 }} />
                        Xuất báo cáo
                    </button>
                    <button className="header-icon-btn">
                        <span className="material-symbols-outlined">notifications</span>
                    </button>
                </div>
            </header>

            <div className="page-container" style={{ padding: '32px' }}>
                {/* ── Summary Cards ─────────────────────────────────────────────────── */}
                <div className="status-cards-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 24 }}>
                    <div className="card" style={{ padding: '16px 20px' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                            Tổng số sự cố <span className="material-symbols-outlined" style={{ fontSize: 16 }}>list_alt</span>
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{alerts.length}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--color-normal)', marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>trending_down</span> -5% so với tháng trước
                        </div>
                    </div>
                    <div className="card" style={{ padding: '16px 20px', borderBottom: '3px solid var(--color-normal)' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                            Đã xử lý <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--color-normal)' }}>check_circle</span>
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--color-normal)' }}>{resolvedCount}</div>
                    </div>
                    <div className="card" style={{ padding: '16px 20px' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                            Đang chờ xử lý <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--color-critical)' }}>error</span>
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--color-critical)' }}>
                            {unacked.toString().padStart(2, '0')}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>Cần phản hồi ngay lập tức</div>
                    </div>
                    <div className="card" style={{ padding: '16px 20px' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                            Thời gian phản hồi TB <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent-primary)' }}>timer</span>
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>1m 45s</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--color-normal)', marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>trending_down</span> Cải thiện 12%
                        </div>
                    </div>
                </div>

                {/* ── Main Layout: Table + Right Panel ─────────────────────────────── */}
                <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: 24 }}>

                    {/* Left Grid: Incident Table */}
                    <div className="data-table-container">
                        <div className="data-table-header">
                            <h3 className="card-title">
                                <span className="material-symbols-outlined" style={{ color: 'var(--accent-primary)' }}>toc</span>
                                Chi tiết nhật ký vận hành
                            </h3>
                            <div style={{ display: 'flex', gap: 8 }}>
                                <select
                                    value={filterSev}
                                    onChange={e => setFilterSev(e.target.value)}
                                    style={{ padding: '6px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', background: 'var(--bg-base)', outline: 'none' }}
                                >
                                    {SEVERITIES.map(s => <option key={s} value={s}>{s === 'ALL' ? 'Mức độ (Tất cả)' : s}</option>)}
                                </select>
                                <select
                                    value={filterStage}
                                    onChange={e => setFilterStage(e.target.value)}
                                    style={{ padding: '6px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', background: 'var(--bg-base)', outline: 'none' }}
                                >
                                    {STAGES.map(s => <option key={s} value={s}>{s === 'ALL' ? 'Khu vực (Tất cả)' : `Trạm ${s}`}</option>)}
                                </select>
                            </div>
                        </div>

                        {loading ? (
                            <div style={{ padding: 64, textAlign: 'center', color: 'var(--text-muted)' }}>Đang tải dữ liệu...</div>
                        ) : filtered.length === 0 ? (
                            <div style={{ padding: 64, textAlign: 'center', color: 'var(--text-muted)' }}>Không có sự cố nào khớp với bộ lọc.</div>
                        ) : (
                            <div style={{ overflowX: 'auto' }}>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>Thời gian</th>
                                            <th>Khu vực</th>
                                            <th>Mức độ</th>
                                            <th>Trạng thái</th>
                                            <th>Người xử lý</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filtered.map(a => (
                                            <tr key={a.id}>
                                                <td>{formatDateTime(a.timestamp)}</td>
                                                <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                                                    Sự cố {a.stage}<br />
                                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 400 }}>{a.message?.split(' at ')[0] || `Anomaly detected`}</span>
                                                </td>
                                                <td>
                                                    <span className={`severity-badge ${(a.severity || 'LOW').toLowerCase()}`}>
                                                        {a.severity || 'LOW'}
                                                    </span>
                                                </td>
                                                <td>
                                                    {a.acknowledged ? (
                                                        <span className="status-badge success">
                                                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>check_circle</span>
                                                            ĐÃ XỬ LÝ
                                                        </span>
                                                    ) : (
                                                        <button className="btn-outline" style={{ padding: '4px 12px', fontSize: '0.75rem', borderRadius: 999 }} onClick={() => handleAck(a.id)}>
                                                            ĐANG CHỜ
                                                        </button>
                                                    )}
                                                </td>
                                                <td>
                                                    {a.acknowledged ? (
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}>
                                                            <span className="avatar-small">AD</span>
                                                            Admin User
                                                        </div>
                                                    ) : (
                                                        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic' }}>Chưa tiếp nhận</span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                            <span>Hiển thị {filtered.length} trên {alerts.length} sự cố</span>
                            <div style={{ display: 'flex', gap: 4 }}>
                                <button style={{ width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer' }}>&lt;</button>
                                <button style={{ width: 28, height: 28, border: 'none', background: 'var(--accent-primary)', color: '#fff', borderRadius: 4, cursor: 'pointer' }}>1</button>
                                <button style={{ width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer' }}>2</button>
                                <button style={{ width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer' }}>3</button>
                                <button style={{ width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer' }}>&gt;</button>
                            </div>
                        </div>
                    </div>

                    {/* Right Grid: Stats & Charts (Mockup visuals) */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title" style={{ fontSize: '0.85rem' }}>Phân bổ sự cố theo khu vực</h3>
                                <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: 16 }}>bar_chart</span>
                            </div>
                            <div style={{ padding: '20px' }}>
                                {/* Mockup chart bars */}
                                {[
                                    { name: 'Khu clo va tham thau nguoc (P4)', val: 32, p: '100%', color: 'var(--accent-primary)' },
                                    { name: 'Lam sach he thong (P6)', val: 24, p: '75%', color: 'var(--color-medium)' },
                                    { name: 'Xu ly hoa chat (P2)', val: 19, p: '59%', color: 'var(--border-subtle)' },
                                    { name: 'Thu hoi nuoc sach (P5)', val: 14, p: '44%', color: 'var(--border-subtle)' },
                                ].map((bar, i) => (
                                    <div key={i} style={{ marginBottom: 12 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                                            <span>{bar.name}</span>
                                            <span style={{ fontWeight: 600 }}>{bar.val}</span>
                                        </div>
                                        <div style={{ height: 6, background: 'var(--bg-base)', borderRadius: 3, overflow: 'hidden' }}>
                                            <div style={{ width: bar.p, background: bar.color, height: '100%', borderRadius: 3 }}></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title" style={{ fontSize: '0.85rem' }}>Trạng thái cảm biến</h3>
                            </div>
                            <div style={{ padding: '20px', display: 'flex', gap: 16 }}>
                                <div style={{ flex: 1, padding: 16, background: '#f0fdf4', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.75rem', color: '#16a34a', fontWeight: 600, marginBottom: 4 }}>ONLINE</div>
                                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#16a34a' }}>42/42</div>
                                </div>
                                <div style={{ flex: 1, padding: 16, background: 'var(--bg-base)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 4 }}>BATTERY</div>
                                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>98%</div>
                                </div>
                            </div>
                        </div>

                        <div style={{ textAlign: 'center', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                            © 2026 SCADA-SECURITY Hệ thống Giám sát & Quản lý Hạ tầng Nước • v4.2.0-LTS
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}


