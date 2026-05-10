const memoryCache = {};

function canUseSessionStorage() {
    return typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined';
}

function readRaw(key) {
    if (Object.prototype.hasOwnProperty.call(memoryCache, key)) {
        return memoryCache[key];
    }

    if (!canUseSessionStorage()) {
        return null;
    }

    try {
        const value = window.sessionStorage.getItem(key);
        if (value !== null) {
            memoryCache[key] = value;
        }
        return value;
    } catch {
        return null;
    }
}

function writeRaw(key, value) {
    memoryCache[key] = value;

    if (!canUseSessionStorage()) {
        return;
    }

    try {
        window.sessionStorage.setItem(key, value);
    } catch {
        // ignore storage write failures
    }
}

export function readSessionState(key, fallbackValue) {
    const raw = readRaw(key);
    if (!raw) {
        return fallbackValue;
    }

    try {
        return JSON.parse(raw);
    } catch {
        return fallbackValue;
    }
}

export function writeSessionState(key, value) {
    try {
        writeRaw(key, JSON.stringify(value));
    } catch {
        // ignore serialization failures
    }
}
