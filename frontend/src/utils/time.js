const BACKEND_ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}/;
const HAS_TIMEZONE_SUFFIX = /(Z|[+-]\d{2}:?\d{2})$/i;

function normalizeBackendTimestamp(value) {
    if (typeof value !== 'string') {
        return value;
    }

    const trimmed = value.trim();
    if (BACKEND_ISO_TIMESTAMP.test(trimmed) && !HAS_TIMEZONE_SUFFIX.test(trimmed)) {
        return `${trimmed.replace(' ', 'T')}Z`;
    }

    return trimmed;
}

export function parseBackendTimestamp(value) {
    if (value == null) {
        return null;
    }

    if (typeof value === 'number') {
        return Number.isFinite(value) ? value : null;
    }

    const timestamp = Date.parse(normalizeBackendTimestamp(value));
    return Number.isFinite(timestamp) ? timestamp : null;
}

export function getBackendDate(value) {
    const timestamp = parseBackendTimestamp(value);
    if (timestamp == null) {
        return null;
    }

    const date = new Date(timestamp);
    return Number.isNaN(date.getTime()) ? null : date;
}

export function formatBackendTime(value, options = {}) {
    const date = getBackendDate(value);
    return date ? date.toLocaleTimeString('vi-VN', options) : '--';
}

export function formatBackendDate(value, options = {}) {
    const date = getBackendDate(value);
    return date ? date.toLocaleDateString('vi-VN', options) : '--';
}

export function formatBackendDateTime(value, options = {}) {
    const date = getBackendDate(value);
    return date ? date.toLocaleString('vi-VN', options) : '--';
}
