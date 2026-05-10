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
import { STAGE_CONFIG } from '../constants/stages';

const DAY_HISTORY_LIMIT = 86400;
const DEVICE_HISTORY_WINDOW_MS = 60 * 60 * 1000;
const STREAM_GAP_THRESHOLD_MS = 15 * 1000;
const DAY_HISTORY_GAP_THRESHOLD_MS = 5 * 60 * 1000;

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

function getStageCacheKey(stageKey) {
    return `stage-session-${stageKey || 'unknown'}-v1`;
}

function buildDeviceHistory(rows, field, kind) {
    const latestTimestamp = rows.reduce((latest, row) => {
        const ts = parseTimestamp(row.timestamp);
        return ts != null && ts > latest ? ts : latest;
    }, 0);
    const dayStart = latestTimestamp ? latestTimestamp - 24 * 60 * 60 * 1000 : 0;

    return rows
        .map((row) => {
            const timestamp = parseTimestamp(row.timestamp);
            const source = kind === 'actuator' ? row.actuator_values : row.sensor_values;
            const fallbackSource = row.raw_values || {};
            const value = parseNumericValue(source?.[field] ?? fallbackSource[field]);

            if (timestamp == null || value == null || (dayStart && timestamp < dayStart)) {
                return null;
            }

            return {
                ts: timestamp,
                value,
                isAnomaly: Boolean(row.is_anomaly),
            };
        })
        .filter(Boolean);
}

function formatDetailTimestamp(timestamp) {
    const date = new Date(timestamp);
    return Number.isNaN(date.getTime()) ? '--' : date.toLocaleString('vi-VN');
}

function DeviceHistoryModal({ device, points, loading, onClose }) {
    if (!device) {
        return null;
    }

    const latestPoint = points.at(-1);
    const minValue = points.length ? Math.min(...points.map((point) => point.value)) : null;
    const maxValue = points.length ? Math.max(...points.map((point) => point.value)) : null;

    return (
        <div className="device-modal-backdrop" role="presentation" onClick={onClose}>
            <section className="device-modal" role="dialog" aria-modal="true" aria-label={`${device.field} 24 hour history`} onClick={(event) => event.stopPropagation()}>
                <div className="device-modal-header">
                    <div>
                        <div className="device-modal-eyebrow">{device.kind === 'actuator' ? 'Actuator' : 'Sensor'} history</div>
                        <h2 className="device-modal-title">{device.field}</h2>
                        <div className="device-modal-subtitle">Last 24 hours from stored stage history</div>
                    </div>
                    <button className="device-modal-close" type="button" onClick={onClose} aria-label="Close device history">
                        <span className="material-symbols-outlined">close</span>
                    </button>
                </div>

                <div className="device-modal-stats">
                    <div>
                        <span>Latest</span>
                        <strong>{latestPoint ? latestPoint.value.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                        <span>Min</span>
                        <strong>{minValue != null ? minValue.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                        <span>Max</span>
                        <strong>{maxValue != null ? maxValue.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                        <span>Samples</span>
                        <strong>{points.length}</strong>
                    </div>
                </div>

                <div className="device-modal-chart">
                    {loading ? (
                        <div className="device-modal-empty">Loading device history...</div>
                    ) : points.length > 0 ? (
                        <SensorChart
                            data={points}
                            threshold={null}
                            height={300}
                            dataKey="value"
                            windowMs={DEVICE_HISTORY_WINDOW_MS}
                            tickStepMs={60 * 60 * 1000}
                            showMiniOverview
                            resetKey={`modal-${device.field}`}
                            gapThresholdMs={DAY_HISTORY_GAP_THRESHOLD_MS}
                        />
                    ) : (
                        <div className="device-modal-empty">No stored values for this device</div>
                    )}
                </div>

                <div className="device-modal-table-wrap">
                    <table className="device-modal-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Value</th>
                                <th>Window state</th>
                            </tr>
                        </thead>
                        <tbody>
                            {points.slice(-300).reverse().map((point) => (
                                <tr key={`${point.ts}-${point.value}`}>
                                    <td>{formatDetailTimestamp(point.ts)}</td>
                                    <td>{point.value.toFixed(4)}</td>
                                    <td>{point.isAnomaly ? 'DANGER' : 'NORMAL'}</td>
                                </tr>
                            ))}
                            {!loading && points.length === 0 && (
                                <tr>
                                    <td colSpan="3">No rows available</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
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
                                onClick={() => openDeviceHistory({ field: sensor, kind: 'sensor' })}
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
                                    legendLabel={sensor}
                                    latestValue={currentData[sensor]}
                                    resetKey={`${stageKey}-${sensor}`}
                                    gapThresholdMs={STREAM_GAP_THRESHOLD_MS}
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
                                    onClick={() => openDeviceHistory({ field: actuator, kind: 'actuator' })}
                                    onKeyDown={(event) => {
                                        if (event.key === 'Enter' || event.key === ' ') {
                                            event.preventDefault();
                                            openDeviceHistory({ field: actuator, kind: 'actuator' });
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
