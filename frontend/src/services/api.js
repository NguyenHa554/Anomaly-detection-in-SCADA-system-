// ─── API base URL ────────────────────────────────────────────────────────────
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─── Generic fetch wrapper ────────────────────────────────────────────────────
async function request(path, options = {}) {
    const url = `${BASE_URL}${path}`;
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
    });
    if (!res.ok) {
        const text = await res.text().catch(() => res.statusText);
        throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

/** GET /api/status → { model_loaded,  loaded_stages, stages, server_time} */
export const getStatus = () => request('/api/status');

/** GET /api/alerts */
export const getAlerts = (params = {}) => {
    const qs = new URLSearchParams(
        Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== ''))
    ).toString();
    return request(`/api/alerts${qs ? `?${qs}` : ''}`);
};

/** POST /api/alerts/:id/acknowledge */
export const acknowledgeAlert = (id) =>
    request(`/api/alerts/${id}/acknowledge`, { method: 'POST' });

/** GET /api/incidents */
export const getIncidents = (params = {}) => {
    const qs = new URLSearchParams(
        Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== ''))
    ).toString();
    return request(`/api/incidents${qs ? `?${qs}` : ''}`);
};

/** POST /api/incidents/:id/acknowledge */
export const acknowledgeIncident = (id) =>
    request(`/api/incidents/${id}/acknowledge`, { method: 'POST' });

/** GET /api/history?stage=P1&limit=200 */
export const getHistory = (stageOrOptions, limit = 200) => {
    const options = typeof stageOrOptions === 'object' && stageOrOptions !== null
        ? stageOrOptions
        : { stage: stageOrOptions, limit };
    const qs = new URLSearchParams(
        Object.fromEntries(
            Object.entries(options).filter(([, value]) => value !== undefined && value !== null && value !== '')
        )
    ).toString();
    return request(`/api/history${qs ? `?${qs}` : ''}`);
};

/** POST /api/ingest — single row prediction */
export const ingestRow = (data) =>
    request('/api/ingest', { method: 'POST', body: JSON.stringify(data) });

/** POST /api/runtime/reload */
export const reloadRuntime = () =>
    request('/api/runtime/reload', { method: 'POST' });

/** POST /api/runtime/reset */
export const resetRuntime = () =>
    request('/api/runtime/reset', { method: 'POST' });

/** POST /api/upload-csv — multipart form */
export const uploadCsv = async (file) => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE_URL}/api/upload-csv`, {
        method: 'POST',
        body: form,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json();
};

export { BASE_URL };
