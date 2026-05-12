import { readSessionState, writeSessionState } from './uiSessionCache';

export const CHART_WINDOW_MS = 60 * 1000;
export const CHART_HISTORY_MS = 30 * 60 * 1000;
export const BACKFILL_HISTORY_LIMIT = 5000;
export const MAX_STORED_POINTS = 5000;
export const DEFAULT_CHART_TICK_STEP_MS = 5000;
export const MIN_CHART_TICK_PX = 86;

const NICE_INTERVALS_MS = [
    1000,
    2000,
    5000,
    10_000,
    15_000,
    30_000,
    60_000,
    2 * 60_000,
    5 * 60_000,
    10 * 60_000,
    15 * 60_000,
    30 * 60_000,
    60 * 60_000,
];

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

export function resolveChartTickStepMs({
    windowMs = CHART_WINDOW_MS,
    width = 0,
    preferredStepMs = DEFAULT_CHART_TICK_STEP_MS,
    minTickPx = MIN_CHART_TICK_PX,
} = {}) {
    if (!Number.isFinite(windowMs) || windowMs <= 0) {
        return preferredStepMs;
    }

    const visibleWidth = Number.isFinite(width) && width > 0 ? width : 420;
    const maxTicks = Math.max(3, Math.floor(visibleWidth / minTickPx));
    const minimumStep = Math.ceil(windowMs / Math.max(1, maxTicks - 1));
    const targetStep = Math.max(preferredStepMs || 0, minimumStep);

    return NICE_INTERVALS_MS.find((step) => step >= targetStep) || targetStep;
}

export function aggregateSeriesByTime(series = [], {
    dataKey = 'value',
    intervalMs = 1000,
    mode = 'latest',
} = {}) {
    if (!intervalMs || intervalMs <= 1000 || series.length <= 2) {
        return series;
    }

    const buckets = new Map();

    series.forEach((point) => {
        const value = Number(point?.[dataKey]);
        if (!Number.isFinite(point?.ts) || !Number.isFinite(value)) {
            return;
        }

        const bucketTs = Math.floor(point.ts / intervalMs) * intervalMs;
        const bucket = buckets.get(bucketTs) || {
            ts: bucketTs,
            count: 0,
            sum: 0,
            min: value,
            max: value,
            latest: value,
            latestTs: point.ts,
            isAnomaly: false,
        };

        bucket.count += 1;
        bucket.sum += value;
        bucket.min = Math.min(bucket.min, value);
        bucket.max = Math.max(bucket.max, value);
        bucket.isAnomaly = bucket.isAnomaly || Boolean(point.isAnomaly);

        if (point.ts >= bucket.latestTs) {
            bucket.latest = value;
            bucket.latestTs = point.ts;
        }

        buckets.set(bucketTs, bucket);
    });

    return Array.from(buckets.values())
        .sort((left, right) => left.ts - right.ts)
        .map((bucket) => {
            const value = mode === 'avg'
                ? bucket.sum / bucket.count
                : mode === 'max'
                    ? bucket.max
                    : mode === 'min'
                        ? bucket.min
                        : bucket.latest;

            return {
                ts: bucket.latestTs,
                [dataKey]: value,
                isAnomaly: bucket.isAnomaly,
                samples: bucket.count,
            };
        });
}
