import { useCallback, useEffect, useMemo, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getAlerts, getStatus } from '../services/api';
import { readSessionState, writeSessionState } from '../services/uiSessionCache';
import {
    appendSeriesPoint,
    CHART_HISTORY_MS,
    writeChartSeriesStore,
} from '../services/chartSeriesStore';
import { STAGE_CONFIG, STAGES } from '../constants/stages';
import { ScadaStreamContext } from './scadaStreamContextValue';
const DASHBOARD_CACHE_KEY = 'dashboard-session-v2';
const EMPTY_CHART_DATA = Object.fromEntries(STAGES.map((stage) => [stage, []]));
const EMPTY_SCORES = Object.fromEntries(STAGES.map((stage) => [stage, null]));
const EMPTY_ANOMALY_STATES = Object.fromEntries(STAGES.map((stage) => [stage, false]));
const INITIAL_WARMING_STATES = Object.fromEntries(
    STAGES.map((stage) => [stage, Boolean(STAGE_CONFIG[stage]?.monitored)])
);

function getStageCacheKey(stageKey) {
    return `stage-session-${stageKey || 'unknown'}-v1`;
}

function parseTimestamp(value) {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? timestamp : null;
}

function parseNumericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function buildSeriesPoint(timestamp, value, isAnomaly) {
    const ts = parseTimestamp(timestamp);
    if (ts == null || value == null) {
        return null;
    }

    return { ts, value, isAnomaly: Boolean(isAnomaly) };
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

function buildStageUiStatus(config, stageStatus) {
    if (!config?.monitored) {
        return {
            mode: 'normal',
            message: 'NOT AI MONITORED',
            detail: 'This stage remains visible for sensors and actuators only.',
        };
    }

    if (!stageStatus || stageStatus.status === 'warming_up' || stageStatus.ready === false) {
        const bufferFill = stageStatus?.buffer_fill ?? 0;
        const bufferNeeded = stageStatus?.buffer_needed ?? 0;
        return {
            mode: 'warming',
            message: 'WARMING UP',
            detail: `Buffered samples: ${bufferFill}/${bufferNeeded}`,
        };
    }

    const isDanger = Boolean(stageStatus.is_anomaly);
    return {
        mode: isDanger ? 'danger' : 'normal',
        message: isDanger ? 'DANGER' : 'NORMAL',
        detail: isDanger
            ? 'Confirmed anomaly window detected for this stage.'
            : 'The stage is within the learned normal window.',
    };
}

function readDashboardCache() {
    const cached = readSessionState(DASHBOARD_CACHE_KEY, null);
    return {
        status: cached?.status || null,
        chartData: { ...EMPTY_CHART_DATA, ...(cached?.chartData || {}) },
        scores: { ...EMPTY_SCORES, ...(cached?.scores || {}) },
        anomalyStates: { ...EMPTY_ANOMALY_STATES, ...(cached?.anomalyStates || {}) },
        warmingStates: { ...INITIAL_WARMING_STATES, ...(cached?.warmingStates || {}) },
        alerts: Array.isArray(cached?.alerts) ? cached.alerts : [],
    };
}

function readStageCaches() {
    return STAGES.reduce((acc, stage) => {
        const cached = readSessionState(getStageCacheKey(stage), null);
        acc.chartData[stage] = cached?.chartData || {};
        acc.currentData[stage] = cached?.currentData || {};
        acc.statuses[stage] = cached?.status || {
            mode: STAGE_CONFIG[stage]?.monitored ? 'warming' : 'normal',
            message: STAGE_CONFIG[stage]?.monitored ? 'WARMING UP' : 'NOT AI MONITORED',
            detail: STAGE_CONFIG[stage]?.monitored
                ? 'The model is collecting enough samples to evaluate the window.'
                : 'This stage remains visible for sensors and actuators only.',
        };
        return acc;
    }, { chartData: {}, currentData: {}, statuses: {} });
}

export function ScadaStreamProvider({ children }) {
    const cachedDashboard = useMemo(() => readDashboardCache(), []);
    const cachedStages = useMemo(() => readStageCaches(), []);
    const [status, setStatus] = useState(cachedDashboard.status);
    const [dashboardChartData, setDashboardChartData] = useState(cachedDashboard.chartData);
    const [scores, setScores] = useState(cachedDashboard.scores);
    const [anomalyStates, setAnomalyStates] = useState(cachedDashboard.anomalyStates);
    const [warmingStates, setWarmingStates] = useState(cachedDashboard.warmingStates);
    const [alerts, setAlerts] = useState(cachedDashboard.alerts);
    const [stageChartData, setStageChartData] = useState(cachedStages.chartData);
    const [stageCurrentData, setStageCurrentData] = useState(cachedStages.currentData);
    const [stageStatuses, setStageStatuses] = useState(cachedStages.statuses);

    const applyRuntimeStatus = useCallback((runtimeStatus) => {
        setStatus(runtimeStatus);
        if (!runtimeStatus?.stages) return;

        setWarmingStates((prev) => {
            const next = { ...prev };
            STAGES.forEach((stage) => {
                const stageStatus = runtimeStatus.stages?.[stage];
                if (stageStatus) {
                    next[stage] = !stageStatus.ready;
                }
            });
            return next;
        });

        setStageStatuses((prev) => {
            const next = { ...prev };
            STAGES.forEach((stage) => {
                const stageStatus = runtimeStatus.stages?.[stage];
                if (stageStatus) {
                    next[stage] = buildStageUiStatus(STAGE_CONFIG[stage], stageStatus);
                }
            });
            return next;
        });
    }, []);

    const handleMessage = useCallback((msg) => {
        if (msg.type === 'sensor_update') {
            const { stages = [] } = msg;
            const rawData = msg.raw_data || {};
            const nextScores = {};
            const nextAnomalyStates = {};
            const nextWarmingStates = {};
            const nextStageStatuses = {};
            const nextStageCurrentData = {};

            stages.forEach((stageStatus) => {
                const stageKey = stageStatus.stage;
                const score = Number.isFinite(stageStatus.max_z_score) ? stageStatus.max_z_score : null;
                nextScores[stageKey] = score;
                nextAnomalyStates[stageKey] = Boolean(stageStatus.is_anomaly);
                nextWarmingStates[stageKey] = stageStatus.status === 'warming_up';
                nextStageStatuses[stageKey] = buildStageUiStatus(STAGE_CONFIG[stageKey], stageStatus);
                nextStageCurrentData[stageKey] = buildCurrentData(
                    STAGE_CONFIG[stageKey] || {},
                    stageStatus.sensor_values || {},
                    stageStatus.actuator_values || {},
                    rawData
                );
            });

            setScores((prev) => ({ ...prev, ...nextScores }));
            setAnomalyStates((prev) => ({ ...prev, ...nextAnomalyStates }));
            setWarmingStates((prev) => ({ ...prev, ...nextWarmingStates }));
            setStageStatuses((prev) => ({ ...prev, ...nextStageStatuses }));
            setStageCurrentData((prev) => ({ ...prev, ...nextStageCurrentData }));

            setDashboardChartData((prev) => {
                const next = { ...prev };
                stages.forEach((stageStatus) => {
                    const score = Number.isFinite(stageStatus.max_z_score) ? stageStatus.max_z_score : null;
                    const ts = parseTimestamp(stageStatus.timestamp || msg.timestamp);
                    if (score == null || ts == null) return;
                    next[stageStatus.stage] = appendSeriesPoint(next[stageStatus.stage] || [], {
                        ts,
                        score,
                        isAnomaly: Boolean(stageStatus.is_anomaly),
                    }, CHART_HISTORY_MS);
                });
                writeChartSeriesStore(DASHBOARD_CACHE_KEY, { chartData: next });
                return next;
            });

            setStageChartData((prev) => {
                const next = { ...prev };
                stages.forEach((stageStatus) => {
                    const stageKey = stageStatus.stage;
                    const config = STAGE_CONFIG[stageKey];
                    if (!config) return;

                    const stageCharts = { ...(next[stageKey] || {}) };
                    const messageTimestamp = stageStatus.timestamp || msg.timestamp;
                    (config.sensors || []).forEach((sensor) => {
                        const value = parseNumericValue(stageStatus.sensor_values?.[sensor] ?? rawData[sensor]);
                        const point = buildSeriesPoint(messageTimestamp, value, stageStatus.is_anomaly);
                        if (!point) return;
                        stageCharts[sensor] = appendSeriesPoint(stageCharts[sensor] || [], point, CHART_HISTORY_MS);
                    });
                    next[stageKey] = stageCharts;
                    writeChartSeriesStore(getStageCacheKey(stageKey), {
                        chartData: stageCharts,
                        currentData: nextStageCurrentData[stageKey],
                        status: nextStageStatuses[stageKey],
                    });
                });
                return next;
            });
        }

        if (msg.type === 'alert' && msg.alert) {
            setAlerts((prev) => [msg.alert, ...prev].slice(0, 100));
        }

        if (msg.type === 'status') {
            applyRuntimeStatus(msg);
        }
    }, [applyRuntimeStatus]);

    const { connected } = useWebSocket({ onMessage: handleMessage });

    useEffect(() => {
        writeSessionState(DASHBOARD_CACHE_KEY, {
            status,
            chartData: dashboardChartData,
            scores,
            anomalyStates,
            warmingStates,
            alerts,
        });
    }, [alerts, anomalyStates, dashboardChartData, scores, status, warmingStates]);

    useEffect(() => {
        getStatus().then(applyRuntimeStatus).catch(() => {});
        getAlerts({ limit: 30 }).then((data) => setAlerts(data.alerts || data)).catch(() => {});
    }, [applyRuntimeStatus]);

    useEffect(() => {
        if (!connected) return undefined;
        const id = setInterval(() => {
            getStatus().then(applyRuntimeStatus).catch(() => {});
        }, 5000);
        return () => clearInterval(id);
    }, [applyRuntimeStatus, connected]);

    const value = useMemo(() => ({
        alerts,
        anomalyStates,
        applyRuntimeStatus,
        connected,
        dashboardChartData,
        scores,
        setAlerts,
        setAnomalyStates,
        setDashboardChartData,
        setScores,
        setStageChartData,
        setStageCurrentData,
        setStageStatuses,
        setWarmingStates,
        stageChartData,
        stageCurrentData,
        stageStatuses,
        status,
        warmingStates,
    }), [
        alerts,
        anomalyStates,
        applyRuntimeStatus,
        connected,
        dashboardChartData,
        scores,
        stageChartData,
        stageCurrentData,
        stageStatuses,
        status,
        warmingStates,
    ]);

    return (
        <ScadaStreamContext.Provider value={value}>
            {children}
        </ScadaStreamContext.Provider>
    );
}
