import { useState, useEffect, useCallback } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { getHistory, getStatus } from '../services/api';
import {
    BACKFILL_HISTORY_LIMIT,
    CHART_WINDOW_MS,
    mergeSeries,
    normalizeSeries,
    writeChartSeriesStore,
} from '../services/chartSeriesStore';
import { useScadaStream } from '../context/scadaStreamContextValue';
import StatusCard from '../components/StatusCard';
import SensorChart from '../components/SensorChart';
import DeviceHistoryModal from '../components/DeviceHistoryModal';
import {
    buildDeviceHistory,
    DAY_HISTORY_LIMIT,
} from '../services/deviceHistory';
import { STAGE_CONFIG } from '../constants/stages';
import { parseBackendTimestamp } from '../utils/time';

const STREAM_GAP_THRESHOLD_MS = 15 * 1000;

function parseNumericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function parseTimestamp(value) {
    return parseBackendTimestamp(value);
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

function getStageCacheKey(stageKey) {
    return `stage-session-${stageKey || 'unknown'}-v1`;
}

export default function StagePage() {
    const { stageId } = useParams();
    const stageKey = stageId?.toUpperCase();
    const config = STAGE_CONFIG[stageKey];
    const cacheKey = getStageCacheKey(stageKey);
    const {
        applyRuntimeStatus,
        setStageChartData,
        setStageCurrentData,
        setStageStatuses,
        stageChartData,
        stageCurrentData,
        stageStatuses,
    } = useScadaStream();
    const chartData = stageChartData[stageKey] || {};
    const currentData = stageCurrentData[stageKey] || {};
    const status = stageStatuses[stageKey] || {
        mode: config?.monitored ? 'warming' : 'normal',
        message: config?.monitored ? 'WARMING UP' : 'NOT AI MONITORED',
        detail: config?.monitored
            ? 'The model is collecting enough samples to evaluate the window.'
            : 'This stage remains visible for sensors and actuators only.',
    };
    const [selectedDevice, setSelectedDevice] = useState(null);
    const [deviceHistory, setDeviceHistory] = useState([]);
    const [deviceHistoryLoading, setDeviceHistoryLoading] = useState(false);

    const openDeviceHistory = useCallback((device) => {
        setDeviceHistory([]);
        setDeviceHistoryLoading(true);
        setSelectedDevice(device);
    }, []);

    const closeDeviceHistory = useCallback(() => {
        setSelectedDevice(null);
        setDeviceHistory([]);
        setDeviceHistoryLoading(false);
    }, []);

    useEffect(() => {
        if (!selectedDevice || !stageKey) {
            return;
        }

        let cancelled = false;
        getHistory({ stage: stageKey, limit: DAY_HISTORY_LIMIT }).then((rows) => {
            if (cancelled) return;
            setDeviceHistory(buildDeviceHistory(rows, selectedDevice.field, selectedDevice.kind));
        }).catch(() => {
            if (!cancelled) setDeviceHistory([]);
        }).finally(() => {
            if (!cancelled) setDeviceHistoryLoading(false);
        });

        return () => {
            cancelled = true;
        };
    }, [selectedDevice, stageKey]);

    useEffect(() => {
        if (!config) {
            return;
        }

        getHistory({ stage: stageKey, limit: BACKFILL_HISTORY_LIMIT }).then((rows) => {
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
                    nextChartData[sensor] = normalizeSeries(series);
                });
            });

            Object.keys(nextChartData).forEach((sensor) => {
                nextChartData[sensor] = normalizeSeries(nextChartData[sensor]);
            });

            setStageChartData((prev) => {
                const previousStageData = prev[stageKey] || {};
                const merged = { ...prev };
                const mergedStageData = { ...previousStageData };
                Object.keys(nextChartData).forEach((sensor) => {
                    mergedStageData[sensor] = normalizeSeries(
                        mergeSeries(previousStageData[sensor] || [], nextChartData[sensor] || [])
                    );
                });
                merged[stageKey] = mergedStageData;
                writeChartSeriesStore(cacheKey, { chartData: mergedStageData });
                return merged;
            });

            const latestRow = rows.at(-1);
            if (latestRow) {
                const nextCurrentData = buildCurrentData(
                    config,
                    latestRow.sensor_values || {},
                    latestRow.actuator_values || {},
                    latestRow.raw_values || {}
                );
                setStageCurrentData((prev) => ({ ...prev, [stageKey]: nextCurrentData }));

                if (config.monitored) {
                    setStageStatuses((prev) => {
                        if (prev[stageKey]?.mode === 'warming') {
                            return prev;
                        }

                        return {
                            ...prev,
                            [stageKey]: {
                            mode: latestRow.is_anomaly ? 'danger' : 'normal',
                            message: latestRow.is_anomaly ? 'DANGER' : 'NORMAL',
                            detail: latestRow.is_anomaly
                                ? 'Confirmed anomaly window detected for this stage.'
                                : 'The stage is within the learned normal window.',
                            },
                        };
                    });
                }
            }
        }).catch(() => {});

        getStatus().then((runtimeStatus) => {
            applyRuntimeStatus(runtimeStatus);
        }).catch(() => {});
    }, [applyRuntimeStatus, cacheKey, config, setStageChartData, setStageCurrentData, setStageStatuses, stageKey]);

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
                            <button
                                key={sensor}
                                type="button"
                                className="device-chart-card"
                                onClick={() => openDeviceHistory({ field: sensor, kind: 'sensor', stage: stageKey })}
                            >
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
                                    height={170}
                                    dataKey="value"
                                    windowMs={CHART_WINDOW_MS}
                                    tickStepMs={10000}
                                    minTickPx={96}
                                    legendLabel={sensor}
                                    latestValue={currentData[sensor]}
                                    resetKey={`${stageKey}-${sensor}`}
                                    gapThresholdMs={STREAM_GAP_THRESHOLD_MS}
                                    aggregation="latest"
                                />
                            </button>
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
                                    role="button"
                                    tabIndex={0}
                                    onClick={() => openDeviceHistory({ field: actuator, kind: 'actuator', stage: stageKey })}
                                    onKeyDown={(event) => {
                                        if (event.key === 'Enter' || event.key === ' ') {
                                            event.preventDefault();
                                            openDeviceHistory({ field: actuator, kind: 'actuator', stage: stageKey });
                                        }
                                    }}
                                    className="device-actuator-card"
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

            <DeviceHistoryModal
                device={selectedDevice}
                points={deviceHistory}
                loading={deviceHistoryLoading}
                onClose={closeDeviceHistory}
            />
        </div>
    );
}
