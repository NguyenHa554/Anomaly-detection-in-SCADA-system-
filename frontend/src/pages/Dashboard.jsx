import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getStatus, getAlerts } from '../services/api';
import StatusCard from '../components/StatusCard';
import StageIndicator from '../components/StageIndicator';
import SensorChart from '../components/SensorChart';
import AlertPanel from '../components/AlertPanel';

const STAGE_CONFIG = {
    P1: { threshold: 2.0, window: 30, name: 'Cấp nước thô' },
    P2: { threshold: 2.0, window: 300, name: 'Xử lý hóa chất' },
    P3: { threshold: 2.0, window: 30, name: 'Siêu lọc' },
    P4: { threshold: 2.0, window: 30, name: 'Khử clo và thẩm thấu ngược' },
    P5: { threshold: 2.0, window: 30, name: 'Thu hồi nước sạch' },
    P6: { threshold: 2.0, window: 30, name: 'Làm sạch' },
};
const STAGES = Object.keys(STAGE_CONFIG);
const MAX_CHART_POINTS = 120;

export default function Dashboard() {
    const [status, setStatus] = useState(null);
    const [chartData, setChartData] = useState(
        Object.fromEntries(STAGES.map(s => [s, []]))
    );
    const [scores, setScores] = useState(Object.fromEntries(STAGES.map(s => [s, null])));
    const [alerts, setAlerts] = useState([]);
    const tickRef = useRef(0);

    const handleMessage = useCallback((msg) => {
        if (msg.type === 'sensor_update') {
            const { stages } = msg;
            tickRef.current += 1;
            const t = tickRef.current;
            
            const newScores = {};
            stages.forEach(s => {
                newScores[s.stage] = s.max_z_score;
            });
            setScores(prev => ({ ...prev, ...newScores }));
            
            setChartData(prev => {
                const next = { ...prev };
                stages.forEach(s => {
                    if (!next[s.stage]) next[s.stage] = [];
                    const arr = [...next[s.stage], { t, score: s.max_z_score }];
                    next[s.stage] = arr.slice(-MAX_CHART_POINTS);
                });
                return next;
            });
        }

        if (msg.type === 'alert' && msg.alert) {
            setAlerts(prev => [msg.alert, ...prev].slice(0, 100));
        }

        if (msg.type === 'status') {
            setStatus(msg);
        }
    }, []);

    const { connected } = useWebSocket({ onMessage: handleMessage });

    useEffect(() => {
        getStatus().then(setStatus).catch(() => { });
        getAlerts({ limit: 30 }).then(d => setAlerts(d.alerts || d)).catch(() => { });
    }, []);

    useEffect(() => {
        if (!connected) return;
        const id = setInterval(() => {
            getStatus().then(setStatus).catch(() => { });
        }, 5000);
        return () => clearInterval(id);
    }, [connected]);

    const handleAcknowledged = useCallback((id) => {
        setAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true } : a));
    }, []);

    const activeAlerts = alerts.filter(a => !a.acknowledged).length;
    // Derive anomaly state and stage logic
    const anomalousStages = STAGES.filter(s => scores[s] != null && scores[s] >= STAGE_CONFIG[s].threshold);
    const hasAnomaly = anomalousStages.length > 0;

    const displaySystemStatus = hasAnomaly ? 'ĐANG CÓ SỰ CỐ' : 'BÌNH THƯỜNG';
    const displayStatusColor = hasAnomaly ? 'critical' : 'normal';
    const displayAlertCountStr = activeAlerts > 0 ? `${activeAlerts} Cảnh báo` : 'Không có cảnh báo';
    const displayAlertSub = 'Đang chờ xử lý trong 24h qua';
    const blockedCount = status?.detections_today ?? 12;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Page Header */}
            <header className="page-header">
                <h1 className="page-title">Hệ thống xử lý nước SWaT</h1>
                <div className="header-actions">
                    <div className="search-bar">
                        <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: 18 }}>search</span>
                        <input type="text" placeholder="Tìm kiếm cảm biến..." />
                    </div>
                    <button className="header-icon-btn">
                        <span className="material-symbols-outlined">notifications</span>
                        {activeAlerts > 0 && <span className="notification-badge" />}
                    </button>
                </div>
            </header>

            <div className="page-container">
                {/* ── Status Cards ─────────────────────────────────────────────────── */}
                <div className="status-cards-grid">
                    <StatusCard
                        label="TRẠNG THÁI HỆ THỐNG"
                        value={displaySystemStatus}
                        valColor={displayStatusColor}
                        sub={`Phát hiện bất thường tại Giai đoạn ${anomalousStages[0] || '...'}`}
                        icon="activity_zone"
                        iconColor={hasAnomaly ? 'red' : 'green'}
                    />
                    <StatusCard
                        label="CẢNH BÁO HỆ THỐNG"
                        value={displayAlertCountStr}
                        valColor="neutral"
                        sub={displayAlertSub}
                        icon="warning"
                        iconColor="blue"
                    />
                    <StatusCard
                        label="HIỆU SUẤT VẬN HÀNH"
                        value={`Uptime: 99.8%`}
                        valColor="neutral"
                        sub={<span>Sự cố đã chặn: <strong>{blockedCount}</strong></span>}
                        icon="bolt"
                        iconColor="green"
                    />
                </div>

                {/* ── Stage Monitor ─────────────────────────────────────────────────── */}
                <div className="stage-flow-container">
                    <div className="stage-flow-header">
                        <h2 className="stage-flow-title">
                            <span className="material-symbols-outlined" style={{ color: 'var(--accent-primary)' }}>account_tree</span>
                            Sơ đồ quy trình 6 giai đoạn
                        </h2>
                        <span className="stage-updated-time">
                            Cập nhật lúc: {new Date().toLocaleTimeString('vi-VN')}
                        </span>
                    </div>

                    <div className="stage-nodes-wrapper">
                        {STAGES.map(stage => {
                            const { threshold, name } = STAGE_CONFIG[stage];
                            const score = scores[stage];
                            const isCritical = score != null && score >= threshold;
                            return (
                                <StageIndicator
                                    key={stage}
                                    stage={stage}
                                    name={name}
                                    isCritical={isCritical}
                                />
                            );
                        })}
                    </div>
                </div>

                {/* ── Main content (charts + alerts) ─────────────────────────────── */}
                <div className="dashboard-main">
                    {/* Charts column */}
                    <div className="card">
                        <div className="card-header">
                            <div>
                                <h3 className="card-title">Biểu đồ Áp suất - Giai đoạn P1</h3>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
                                    Cấp nước thô
                                </div>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', fontWeight: 600 }}>
                                <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-primary)' }}></span>
                                Z-Score: {scores['P1'] ? Number(scores['P1']).toFixed(2) : '--'}
                            </div>
                        </div>
                        <SensorChart
                            data={chartData['P1'] || []}
                            threshold={STAGE_CONFIG['P1'].threshold}
                            height={250}
                        />
                    </div>

                    {/* Alert panel column */}
                    <div className="card incident-panel">
                        <div className="card-header">
                            <h3 className="card-title">
                                <span className="material-symbols-outlined" style={{ color: 'var(--color-critical)' }}>history</span>
                                Nhật ký Cảnh báo
                            </h3>
                        </div>
                        <AlertPanel alerts={alerts} onAcknowledged={handleAcknowledged} />
                    </div>
                </div>
            </div>
        </div>
    );
}
