import { useState, useEffect, useCallback } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { useWebSocket } from '../hooks/useWebSocket';
import { getHistory, getStatus } from '../services/api';
import StatusCard from '../components/StatusCard';
import SensorChart from '../components/SensorChart';
import { STAGE_CONFIG } from '../constants/stages';

const MAX_CHART_POINTS = 120;
const HISTORY_GAP_RESET_MS = 15000;

function parseNumericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function parseTimestamp(value) {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? timestamp : null;
}

function buildCurrentData(config, sensorValues = {}, actuatorValues = {}, rawValues = {}) {
    const next = {};

    [...(config.sensors || []), ...(config.actuators || [])].forEach((field) => {
        const value = field in sensorValues
            ? sensorValues[field]
            : field in actuatorValues
                ? actuatorValues[field]
                : rawValues[field];
        next[field] = parseNumericValue(value);
    });

    return next;
}

function buildSeriesPoint(timestamp, value, isAnomaly) {
    const ts = parseTimestamp(timestamp);
    if (ts == null || value == null) {
        return null;
    }

    return { ts, value, isAnomaly: Boolean(isAnomaly) };
}

function appendLivePoint(series, point) {
    const previousPoint = series.at(-1);
    const nextSeries = previousPoint && point.ts - previousPoint.ts > HISTORY_GAP_RESET_MS
        ? [point]
        : [...series, point];
    return nextSeries.slice(-MAX_CHART_POINTS);
}

function trimToRecentWindow(series) {
    if (series.length <= 1) {
        return series;
    }

    let startIndex = series.length - 1;
    while (startIndex > 0) {
        const gap = series[startIndex].ts - series[startIndex - 1].ts;
        if (gap > HISTORY_GAP_RESET_MS) {
            break;
        }
        startIndex -= 1;
    }

    return series.slice(startIndex).slice(-MAX_CHART_POINTS);
}

export default function StagePage() {
    const { stageId } = useParams();
    const stageKey = stageId?.toUpperCase();
    const config = STAGE_CONFIG[stageKey];

    const [chartData, setChartData] = useState({});
    const [currentData, setCurrentData] = useState({});
    const [status, setStatus] = useState({
        mode: 'warming',
        message: 'WARMING UP',
        detail: 'The model is collecting enough samples to evaluate the window.',
    });

    const updateStatusFromStage = useCallback((stageStatus) => {
        if (!config?.monitored) {
            setStatus({
                mode: 'normal',
                message: 'NOT AI MONITORED',
                detail: 'This stage remains visible for sensors and actuators only.',
            });
            return;
        }

        if (!stageStatus || stageStatus.status === 'warming_up' || stageStatus.ready === false) {
            const bufferFill = stageStatus?.buffer_fill ?? 0;
            const bufferNeeded = stageStatus?.buffer_needed ?? 0;
            setStatus({
                mode: 'warming',
                message: 'WARMING UP',
                detail: `Buffered samples: ${bufferFill}/${bufferNeeded}`,
            });
            return;
        }

        const isDanger = Boolean(stageStatus.is_anomaly);
        setStatus({
            mode: isDanger ? 'danger' : 'normal',
            message: isDanger ? 'DANGER' : 'NORMAL',
            detail: isDanger
                ? 'Confirmed anomaly window detected for this stage.'
                : 'The stage is within the learned normal window.',
        });
    }, [config]);

    useEffect(() => {
        if (!config) {
            return;
        }

        getHistory({ stage: stageKey, limit: MAX_CHART_POINTS }).then((rows) => {
            const nextChartData = {};

            rows.forEach((row) => {
                (config.sensors || []).forEach((sensor) => {
                    const value = parseNumericValue(
                        row.sensor_values?.[sensor] ?? row.raw_values?.[sensor]
                    );
                    const point = buildSeriesPoint(row.timestamp, value, row.is_anomaly);
                    if (!point) {
                        return;
                    }

                    const series = [...(nextChartData[sensor] || []), point];
                    nextChartData[sensor] = series.slice(-MAX_CHART_POINTS);
                });
            });

            Object.keys(nextChartData).forEach((sensor) => {
                nextChartData[sensor] = trimToRecentWindow(nextChartData[sensor]);
            });

            setChartData(nextChartData);

            const latestRow = rows.at(-1);
            if (latestRow) {
                setCurrentData(
                    buildCurrentData(
                        config,
                        latestRow.sensor_values || {},
                        latestRow.actuator_values || {},
                        latestRow.raw_values || {}
                    )
                );

                if (config.monitored) {
                    setStatus({
                        mode: latestRow.is_anomaly ? 'danger' : 'normal',
                        message: latestRow.is_anomaly ? 'DANGER' : 'NORMAL',
                        detail: latestRow.is_anomaly
                            ? 'Confirmed anomaly window detected for this stage.'
                            : 'The stage is within the learned normal window.',
                    });
                }
            }
        }).catch(() => {});

        getStatus().then((runtimeStatus) => {
            const stageStatus = runtimeStatus?.stages?.[stageKey];
            updateStatusFromStage(stageStatus);
        }).catch(() => {});
    }, [config, stageKey, updateStatusFromStage]);

    const handleMessage = useCallback((msg) => {
        if (msg.type !== 'sensor_update' || !config) return;

        const rawData = msg.raw_data || {};
        const stageStatus = (msg.stages || []).find((stage) => stage.stage === stageKey);
        const liveSensorValues = stageStatus?.sensor_values || {};
        const liveActuatorValues = stageStatus?.actuator_values || {};

        setCurrentData(
            buildCurrentData(config, liveSensorValues, liveActuatorValues, rawData)
        );

        const messageTimestamp = stageStatus?.timestamp || msg.timestamp;
        setChartData((prev) => {
            const next = { ...prev };

            (config.sensors || []).forEach((sensor) => {
                const value = parseNumericValue(
                    liveSensorValues[sensor] ?? rawData[sensor]
                );
                const point = buildSeriesPoint(messageTimestamp, value, stageStatus?.is_anomaly);
                if (!point) {
                    return;
                }

                next[sensor] = appendLivePoint(prev[sensor] || [], point);
            });

            return next;
        });

        updateStatusFromStage(stageStatus);
    }, [config, stageKey, updateStatusFromStage]);

    useWebSocket({ onMessage: handleMessage });

    if (!config) return <Navigate to="/" />;

    const statusColor = status.mode === 'danger' ? 'critical' : status.mode === 'warming' ? 'warning' : 'normal';
    const statusIconColor = status.mode === 'danger' ? 'red' : status.mode === 'warming' ? 'orange' : 'green';

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <header className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="page-title" style={{ color: 'var(--text-muted)' }}>Overview</span>
                    <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--text-muted)' }}>chevron_right</span>
                    <h1 className="page-title">{stageKey} - {config.name}</h1>
                </div>
            </header>

            <div className="page-container">
                <div style={{ marginBottom: 24 }}>
                    <StatusCard
                        label="STAGE STATUS"
                        value={status.message}
                        valColor={statusColor}
                        sub={`${status.detail} | Sensors: ${config.sensors.length} | Actuators: ${config.actuators.length}`}
                        icon="network_check"
                        iconColor={statusIconColor}
                    />
                </div>

                <div className="card" style={{ marginBottom: 24, padding: 24 }}>
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Continuous sensors</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24 }}>
                        {config.sensors.map((sensor) => (
                            <div key={sensor} style={{ border: '1px solid var(--border-subtle)', borderRadius: 8, padding: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                                    <div style={{ fontWeight: 600 }}>{sensor}</div>
                                    <div style={{ color: status.mode === 'danger' ? 'var(--color-critical)' : status.mode === 'warming' ? 'var(--color-medium)' : 'var(--text-primary)', fontWeight: 700 }}>
                                        {currentData[sensor] !== null && currentData[sensor] !== undefined
                                            ? currentData[sensor].toFixed(2)
                                            : '--'}
                                    </div>
                                </div>
                                <SensorChart
                                    data={chartData[sensor] || []}
                                    threshold={null}
                                    height={150}
                                    dataKey="value"
                                />
                            </div>
                        ))}
                    </div>
                </div>

                <div className="card" style={{ padding: 24 }}>
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Actuators</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
                        {config.actuators.map((actuator) => {
                            const value = currentData[actuator];
                            const isActive = value !== null && value !== undefined && value > 0;

                            return (
                                <div
                                    key={actuator}
                                    style={{
                                        border: `1px solid ${isActive ? 'var(--color-normal)' : 'var(--border-subtle)'}`,
                                        borderRadius: 8,
                                        padding: 16,
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 12,
                                        background: isActive ? '#f0fdf4' : 'transparent',
                                    }}
                                >
                                    <div
                                        style={{
                                            width: 12,
                                            height: 12,
                                            borderRadius: '50%',
                                            background: isActive ? 'var(--color-normal)' : 'var(--border-card)',
                                            boxShadow: isActive ? '0 0 8px var(--color-normal)' : 'none',
                                        }}
                                    />
                                    <div>
                                        <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{actuator}</div>
                                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                            Current value: {value !== null && value !== undefined ? value.toFixed(2) : '--'}
                                        </div>
                                    </div>
                                    <div
                                        style={{
                                            marginLeft: 'auto',
                                            fontSize: '0.8rem',
                                            color: isActive ? 'var(--color-normal)' : 'var(--text-muted)',
                                        }}
                                    >
                                        {isActive ? 'ACTIVE' : 'OFF'}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
}
