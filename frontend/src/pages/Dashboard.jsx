import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getStatus, getAlerts } from '../services/api';
import StatusCard from '../components/StatusCard';
import StageIndicator from '../components/StageIndicator';
import SensorChart from '../components/SensorChart';
import AlertPanel from '../components/AlertPanel';
import { STAGE_CONFIG, STAGES, MONITORED_STAGES, MONITORING_ONLY_STAGES } from '../constants/stages';

const MAX_CHART_POINTS = 120;
const INITIAL_WARMING_STATES = Object.fromEntries(
    STAGES.map((stage) => [stage, Boolean(STAGE_CONFIG[stage]?.monitored)])
);

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
    const [warmingStates, setWarmingStates] = useState(INITIAL_WARMING_STATES);
    const [alerts, setAlerts] = useState([]);
    const tickRef = useRef(0);

    const handleMessage = useCallback((msg) => {
        if (msg.type === 'sensor_update') {
            const { stages = [] } = msg;
            tickRef.current += 1;
            const t = tickRef.current;

            const nextScores = {};
            const nextAnomalyStates = {};
            const nextWarmingStates = {};

            stages.forEach((stage) => {
                nextScores[stage.stage] = Number.isFinite(stage.max_z_score) ? stage.max_z_score : null;
                nextAnomalyStates[stage.stage] = Boolean(stage.is_anomaly);
                nextWarmingStates[stage.stage] = stage.status === 'warming_up';
            });

            setScores((prev) => ({ ...prev, ...nextScores }));
            setAnomalyStates((prev) => ({ ...prev, ...nextAnomalyStates }));
            setWarmingStates((prev) => ({ ...prev, ...nextWarmingStates }));

            setChartData((prev) => {
                const next = { ...prev };
                stages.forEach((stage) => {
                    const score = Number.isFinite(stage.max_z_score) ? stage.max_z_score : null;
                    if (score == null) return;
                    const series = [
                        ...(next[stage.stage] || []),
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

    const effectiveWarmingStates = useMemo(() => {
        const next = { ...warmingStates };
        if (!status?.stages) return next;
        MONITORED_STAGES.forEach((stage) => {
            const stageStatus = status.stages?.[stage];
            next[stage] = stageStatus ? !stageStatus.ready : next[stage];
        });
        return next;
    }, [status, warmingStates]);

    const handleAcknowledged = useCallback((id) => {
        setAlerts((prev) => prev.map((alert) => (
            alert.id === id ? { ...alert, acknowledged: true } : alert
        )));
    }, []);

    const activeAlerts = alerts.filter((alert) => !alert.acknowledged).length;
    const anomalousStages = MONITORED_STAGES.filter((stage) => Boolean(anomalyStates[stage]));
    const warmingStages = MONITORED_STAGES.filter(
        (stage) => !anomalyStates[stage] && Boolean(effectiveWarmingStates[stage])
    );
    const hasAnomaly = anomalousStages.length > 0;
    const hasWarming = warmingStages.length > 0;

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
    const liveThreshold = status?.thresholds?.[chartStage]?.T ?? chartStageConfig.threshold;
    const chartScore = scores[chartStage];
    const monitoringOnlyLabel = MONITORING_ONLY_STAGES.join(', ');

    const systemStatusValue = hasAnomaly ? 'DANGER' : hasWarming ? 'WARMING UP' : 'NORMAL';
    const systemStatusColor = hasAnomaly ? 'critical' : hasWarming ? 'warning' : 'normal';
    const systemStatusSub = hasAnomaly
        ? `Confirmed anomaly at stage ${anomalousStages[0] || chartStage}`
        : hasWarming
            ? `Model warming up at stage ${warmingStages[0]}`
            : 'All production alert stages are ready';
    const alertCountValue = activeAlerts > 0 ? `${activeAlerts} alerts` : 'No active alerts';
    const blockedCount = status?.detections_today ?? 12;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <header className="page-header">
                <h1 className="page-title">SWaT Water Treatment System</h1>
                <div className="header-actions">
                    <div className="search-bar">
                        <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: 18 }}>search</span>
                        <input type="text" placeholder="Search sensors..." />
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
                        label="SYSTEM STATUS"
                        value={systemStatusValue}
                        valColor={systemStatusColor}
                        sub={systemStatusSub}
                        icon="activity_zone"
                        iconColor={hasAnomaly ? 'red' : hasWarming ? 'orange' : 'green'}
                    />
                    <StatusCard
                        label="ALERTS"
                        value={alertCountValue}
                        valColor="neutral"
                        sub="Confirmed episode starts in the last 24h"
                        icon="warning"
                        iconColor="blue"
                    />
                    <StatusCard
                        label="OPERATION"
                        value="Uptime: 99.8%"
                        valColor="neutral"
                        sub={<span>Blocked incidents: <strong>{blockedCount}</strong></span>}
                        icon="bolt"
                        iconColor="green"
                    />
                </div>

                <div className="stage-flow-container">
                    <div className="stage-flow-header">
                        <h2 className="stage-flow-title">
                            <span className="material-symbols-outlined" style={{ color: 'var(--accent-primary)' }}>account_tree</span>
                            Operating pipeline
                        </h2>
                        <span className="stage-updated-time">
                            Updated: {new Date().toLocaleTimeString('vi-VN')}
                        </span>
                    </div>
                    <div style={{
                        marginBottom: 16,
                        padding: '10px 14px',
                        borderRadius: 'var(--radius-md)',
                        background: 'var(--bg-base)',
                        color: 'var(--text-secondary)',
                        fontSize: '0.78rem',
                    }}>
                        Final alerts are generated by P2-P5. {monitoringOnlyLabel} remain visible for sensor monitoring and analysis only.
                    </div>

                    <div className="stage-nodes-wrapper">
                        {STAGES.map((stage) => {
                            const { name, monitored } = STAGE_CONFIG[stage];
                            const isCritical = monitored && Boolean(anomalyStates[stage]);
                            const isWarming = monitored && !isCritical && Boolean(effectiveWarmingStates[stage]);

                            return (
                                <StageIndicator
                                    key={stage}
                                    stage={stage}
                                    name={name}
                                    isCritical={isCritical}
                                    isWarming={isWarming}
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
                                <h3 className="card-title">Z-score trend - Stage {chartStage}</h3>
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
                            threshold={liveThreshold}
                            height={250}
                        />
                    </div>

                    <div className="card incident-panel">
                        <div className="card-header">
                            <div>
                                <h3 className="card-title">
                                    <span className="material-symbols-outlined" style={{ color: 'var(--color-critical)' }}>history</span>
                                    Alert log
                                </h3>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
                                    Production incidents only. Monitoring-only stages stay on the dashboard but do not raise final alerts.
                                </div>
                            </div>
                        </div>
                        <AlertPanel alerts={alerts} onAcknowledged={handleAcknowledged} />
                    </div>
                </div>
            </div>
        </div>
    );
}
