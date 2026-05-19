import { useCallback, useMemo } from 'react';
import { CHART_WINDOW_MS } from '../services/chartSeriesStore';
import { useScadaStream } from '../context/scadaStreamContextValue';
import StatusCard from '../components/StatusCard';
import StageIndicator from '../components/StageIndicator';
import SensorChart from '../components/SensorChart';
import AlertPanel from '../components/AlertPanel';
import { STAGE_CONFIG, STAGES, MONITORED_STAGES, MONITORING_ONLY_STAGES } from '../constants/stages';
import { formatBackendTime } from '../utils/time';

export default function Dashboard() {
    const {
        alerts,
        anomalyStates,
        dashboardChartData: chartData,
        scores,
        setAlerts,
        status,
        warmingStates,
    } = useScadaStream();

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
    }, [setAlerts]);

    const activeAlerts = alerts.filter((alert) => !alert.acknowledged).length;
    const anomalousStages = MONITORED_STAGES.filter((stage) => Boolean(anomalyStates[stage]));
    const warmingStages = MONITORED_STAGES.filter(
        (stage) => !anomalyStates[stage] && Boolean(effectiveWarmingStates[stage])
    );
    const hasAnomaly = anomalousStages.length > 0;
    const hasWarming = warmingStages.length > 0;

    const monitoringOnlyLabel = MONITORING_ONLY_STAGES.join(', ');
    const updatedTime = status?.server_time
        ? formatBackendTime(status.server_time)
        : '--';

    const systemStatusValue = hasAnomaly ? 'DANGER' : hasWarming ? 'WARMING UP' : 'NORMAL';
    const systemStatusColor = hasAnomaly ? 'critical' : hasWarming ? 'warning' : 'normal';
    const systemStatusSub = hasAnomaly
        ? `Confirmed anomaly at stage ${anomalousStages[0] || MONITORED_STAGES[0]}`
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
                            Updated: {updatedTime}
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
                    <div className="dashboard-zscore-grid">
                        {MONITORED_STAGES.map((stage) => {
                            const chartStageConfig = STAGE_CONFIG[stage];
                            const liveThreshold = status?.thresholds?.[stage]?.T ?? chartStageConfig.threshold;
                            const chartScore = scores[stage];

                            return (
                                <div className="card dashboard-zscore-card" key={stage}>
                                    <div className="card-header">
                                        <div>
                                            <h3 className="card-title">Line chart - Stage {stage}</h3>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
                                                Realtime - last 1 minute | {chartStageConfig.name}
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', fontWeight: 600 }}>
                                            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-primary)' }}></span>
                                            Z-Score: {chartScore != null ? Number(chartScore).toFixed(2) : '--'}
                                        </div>
                                    </div>
                                    <SensorChart
                                        data={chartData[stage] || []}
                                        threshold={liveThreshold}
                                        height={260}
                                        windowMs={CHART_WINDOW_MS}
                                        tickStepMs={5000}
                                        minTickPx={92}
                                        showMiniOverview
                                        legendLabel={`Z-score ${stage}`}
                                        latestValue={chartScore}
                                        resetKey={`dashboard-${stage}`}
                                        aggregation="max"
                                    />
                                </div>
                            );
                        })}
                    </div>

                    <div className="card incident-panel dashboard-alert-panel">
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

