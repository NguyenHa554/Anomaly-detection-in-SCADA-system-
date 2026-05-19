import { parseBackendTimestamp } from '../utils/time';

export const DAY_HISTORY_LIMIT = 86400;
export const DEVICE_HISTORY_WINDOW_MS = 60 * 60 * 1000;
export const DEVICE_HISTORY_GAP_THRESHOLD_MS = 5 * 60 * 1000;

function parseNumericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

export function buildDeviceHistory(rows, field, kind) {
    const latestTimestamp = rows.reduce((latest, row) => {
        const ts = parseBackendTimestamp(row.timestamp);
        return ts != null && ts > latest ? ts : latest;
    }, 0);
    const historyStart = latestTimestamp ? latestTimestamp - DEVICE_HISTORY_WINDOW_MS : 0;

    return rows
        .map((row) => {
            const timestamp = parseBackendTimestamp(row.timestamp);
            const source = kind === 'actuator' ? row.actuator_values : row.sensor_values;
            const fallbackSource = row.raw_values || {};
            const value = parseNumericValue(source?.[field] ?? fallbackSource[field]);

            if (timestamp == null || value == null || (historyStart && timestamp < historyStart)) {
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
