import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getStatus, getAlerts } from '../services/api';
import StatusCard from '../components/StatusCard';
import StageIndicator from '../components/StageIndicator';
import SensorChart from '../components/SensorChart';
import AlertPanel from '../components/AlertPanel';
import { STAGE_CONFIG, STAGES, MONITORED_STAGES } from '../constants/stages';

const MAX_CHART_POINTS = 120;

export default function Dashboard() {
    const [status, setStatus] = useState(null);
    const [chartData, setChartData] = useState(
        Object.fromEntries(STAGES.map((stage) => [stage, []]))
    );
    const [scores, setScores] = useState(
        Object.fromEntries(STAGES.map((stage) => [stage, null]))
    );
    const [anomalyStates, setAnomalyStates] = useState(
        Object.fromEntries(STAGES.map((stage) => [stage, false]))
    );
    const [alerts, setAlerts] = useState([]);
    const tickRef = useRef(0);

    const handleMessage = useCallback((msg) => {
        if (msg.type === 'sensor_update') {
            const { stages = [] } = msg;
            tickRef.current += 1;
            const t = tickRef.current;

            const newScores = {};
            const newAnomalyStates = {};
            stages.forEach((stage) => {
                newScores[stage.stage] = Number.isFinite(stage.max_z_score) ? stage.max_z_score : null;
                newAnomalyStates[stage.stage] = Boolean(stage.is_anomaly);
            });
            setScores((prev) => ({ ...prev, ...newScores }));
            setAnomalyStates((prev) => ({ ...prev, ...newAnomalyStates }));

            setChartData((prev) => {
                const next = { ...prev };
                stages.forEach((stage) => {
                    const score = Number.isFinite(stage.max_z_score) ? stage.max_z_score : null;
                    if (score == null) return;
                    if (!next[stage.stage]) next[stage.stage] = [];
                    const series = [
                        ...next[stage.stage],
                        { t, score, isAnomaly: Boolean(stage.is_anomaly) },
                    ];
                    next[stage.stage] = series.slice(-MAX_CHART_POINTS);
                });
                return next;
            });
        }

        if (msg.type === 'alert' && msg.alert) {
            setAlerts((prev) => [msg.alert, ...prev].slice(0, 100));
        }

        if (msg.type === 'status') {
            setStatus(msg);
        }
    }, []);

    const { connected } = useWebSocket({ onMessage: handleMessage });

    useEffect(() => {
        getStatus().then(setStatus).catch(() => {});
        getAlerts({ limit: 30 }).then((data) => setAlerts(data.alerts || data)).catch(() => {});
    }, []);

    useEffect(() => {
        if (!connected) return;
        const id = setInterval(() => {
            getStatus().then(setStatus).catch(() => {});
        }, 5000);
        return () => clearInterval(id);
    }, [connected]);

    const handleAcknowledged = useCallback((id) => {
        setAlerts((prev) => prev.map((alert) => (
            alert.id === id ? { ...alert, acknowledged: true } : alert
        )));
    }, []);

    const activeAlerts = alerts.filter((alert) => !alert.acknowledged).length;
    const anomalousStages = MONITORED_STAGES.filter(
        (stage) => Boolean(anomalyStates[stage])
    );
    const hasAnomaly = anomalousStages.length > 0;

    const highestRiskStage = MONITORED_STAGES.reduce((highest, stage) => {
        if (scores[stage] == null) return highest;
        if (!highest) return stage;
        return scores[stage] > scores[highest] ? stage : highest;
    }, null);

    const highestRiskAnomalousStage = anomalousStages.reduce((highest, stage) => {
        if (scores[stage] == null) return highest;
        if (!highest) return stage;
        return scores[stage] > scores[highest] ? stage : highest;
    }, null);

    const chartStage = highestRiskAnomalousStage || highestRiskStage || MONITORED_STAGES[0];
    const chartStageConfig = STAGE_CONFIG[chartStage];
    const chartScore = scores[chartStage];

    const displaySystemStatus = hasAnomaly ? 'ĐANG CÓ SỰ CỐ' : 'BÌNH THƯỜNG';
    const displayStatusColor = hasAnomaly ? 'critical' : 'normal';
    const displayAlertCountStr = activeAlerts > 0 ? `${activeAlerts} Cảnh báo` : 'Không có cảnh báo';
    const displayAlertSub = 'Đang chờ xử lý trong 24h qua';
    const blockedCount = status?.detections_today ?? 12;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
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
                <div className="status-cards-grid">
                    <StatusCard
                        label="TRẠNG THÁI HỆ THỐNG"
                        value={displaySystemStatus}
                        valColor={displayStatusColor}
                        sub={`Phát hiện bất thường tại Giai đoạn ${anomalousStages[0] || chartStage}`}
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
                        value="Uptime: 99.8%"
                        valColor="neutral"
                        sub={<span>Sự cố đã chặn: <strong>{blockedCount}</strong></span>}
                        icon="bolt"
                        iconColor="green"
                    />
                </div>

                <div className="stage-flow-container">
                    <div className="stage-flow-header">
                        <h2 className="stage-flow-title">
                            <span className="material-symbols-outlined" style={{ color: 'var(--accent-primary)' }}>account_tree</span>
                            Sơ đồ quy trình vận hành
                        </h2>
                        <span className="stage-updated-time">
                            Cập nhật lúc: {new Date().toLocaleTimeString('vi-VN')}
                        </span>
                    </div>

                    <div className="stage-nodes-wrapper">
                        {STAGES.map((stage) => {
                            const { name, monitored } = STAGE_CONFIG[stage];
                            const isCritical = monitored && Boolean(anomalyStates[stage]);

                            return (
                                <StageIndicator
                                    key={stage}
                                    stage={stage}
                                    name={name}
                                    isCritical={isCritical}
                                    isMonitored={monitored}
                                />
                            );
                        })}
                    </div>
                </div>

                <div className="dashboard-main">
                    <div className="card">
                        <div className="card-header">
                            <div>
                                <h3 className="card-title">Biểu đồ áp suất - Giai đoạn {chartStage}</h3>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
                                    {chartStageConfig.name}
                                </div>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', fontWeight: 600 }}>
                                <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-primary)' }}></span>
                                Z-Score: {chartScore != null ? Number(chartScore).toFixed(2) : '--'}
                            </div>
                        </div>
                        <SensorChart
                            data={chartData[chartStage] || []}
                            threshold={chartStageConfig.threshold}
                            height={250}
                        />
                    </div>

                    <div className="card incident-panel">
                        <div className="card-header">
                            <h3 className="card-title">
                                <span className="material-symbols-outlined" style={{ color: 'var(--color-critical)' }}>history</span>
                                Nhật ký cảnh báo
                            </h3>
                        </div>
                        <AlertPanel alerts={alerts} onAcknowledged={handleAcknowledged} />
                    </div>
                </div>
            </div>
        </div>
    );
}
