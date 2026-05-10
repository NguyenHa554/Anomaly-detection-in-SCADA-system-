import { readSessionState, writeSessionState } from './uiSessionCache';

export const CHART_WINDOW_MS = 60 * 1000;
export const CHART_HISTORY_MS = 30 * 60 * 1000;
export const BACKFILL_HISTORY_LIMIT = 5000;
export const MAX_STORED_POINTS = 5000;

export function normalizeSeries(series = [], historyMs = CHART_HISTORY_MS) {
    if (series.length <= 1) {
        return series;
    }

    const sortedSeries = series
        .filter((point) => point && Number.isFinite(point.ts))
        .sort((left, right) => left.ts - right.ts);
    const latestTimestamp = sortedSeries.at(-1)?.ts;

    if (!Number.isFinite(latestTimestamp)) {
        return [];
    }

    const earliestTimestamp = latestTimestamp - historyMs;
    return sortedSeries
        .filter((point) => point.ts >= earliestTimestamp)
        .slice(-MAX_STORED_POINTS);
}

export function mergeSeries(existingSeries = [], incomingSeries = [], historyMs = CHART_HISTORY_MS) {
    const pointsByTimestamp = new Map();

    [...existingSeries, ...incomingSeries].forEach((point) => {
        if (point && Number.isFinite(point.ts)) {
            pointsByTimestamp.set(point.ts, point);
        }
    });

    return normalizeSeries(Array.from(pointsByTimestamp.values()), historyMs);
}

export function appendSeriesPoint(series = [], point, historyMs = CHART_HISTORY_MS) {
    return mergeSeries(series, [point], historyMs);
}

export function readChartSeriesStore(key, fallbackValue) {
    return readSessionState(key, fallbackValue);
}

export function writeChartSeriesStore(key, value) {
    const existing = readSessionState(key, {});
    writeSessionState(key, {
        ...(existing || {}),
        ...value,
    });
}
