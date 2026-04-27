import { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, XCircle, Loader, AlertTriangle } from 'lucide-react';
import { uploadCsv } from '../services/api';

const SEVERITY_COLORS = {
    CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#f59e0b', LOW: '#3b82f6',
};

export default function UploadPage() {
    const [dragOver, setDragOver] = useState(false);
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const inputRef = useRef();

    function onDrop(e) {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) pickFile(f);
    }

    function pickFile(f) {
        if (!f.name.endsWith('.csv')) {
            setError('Please select a .csv file.');
            return;
        }
        setFile(f);
        setResults(null);
        setError(null);
    }

    async function handleUpload() {
        if (!file) return;
        setLoading(true);
        setError(null);
        try {
            const data = await uploadCsv(file);
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    function reset() {
        setFile(null);
        setResults(null);
        setError(null);
    }

    const anomalyCount = results?.predictions?.filter(p => p.is_anomaly).length ?? 0;
    const totalCount = results?.predictions?.length ?? 0;

    return (
        <div className="page-container upload-page">
            <h1 className="page-heading">Upload CSV for Analysis</h1>
            <p className="page-subheading">
                Upload a SWaT-format CSV file to run batch anomaly detection across all pipeline stages.
            </p>

            {/* Drop Zone */}
            <div
                className={`drop-zone${dragOver ? ' drag-over' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept=".csv"
                    className="hidden-input"
                    onChange={e => pickFile(e.target.files?.[0])}
                />
                {file ? (
                    <>
                        <div className="drop-zone-icon"><FileText size={48} color="var(--accent-primary)" /></div>
                        <div className="drop-zone-title">{file.name}</div>
                        <div className="drop-zone-sub" onClick={e => e.stopPropagation()}>
                            {(file.size / 1024).toFixed(1)} KB · Click to change file
                        </div>
                    </>
                ) : (
                    <>
                        <div className="drop-zone-icon"><Upload size={48} color="var(--text-muted)" /></div>
                        <div className="drop-zone-title">Drag &amp; drop your CSV here</div>
                        <div className="drop-zone-sub">or click to browse</div>
                        <div className="drop-zone-hint">Accepts SWaT-format CSV files</div>
                    </>
                )}
            </div>

            {/* Error */}
            {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--color-critical)', marginBottom: 16, fontSize: '0.875rem' }}>
                    <XCircle size={16} /> {error}
                </div>
            )}

            {/* Actions */}
            <div style={{ display: 'flex', gap: 12, marginBottom: 28 }}>
                <button
                    className="btn btn-primary"
                    onClick={handleUpload}
                    disabled={!file || loading}
                >
                    {loading ? <><span className="loading-ring" /> Analyzing…</> : <><Upload size={15} /> Run Detection</>}
                </button>
                {(file || results) && (
                    <button className="btn btn-secondary" onClick={reset}>
                        Clear
                    </button>
                )}
            </div>

            {/* Results */}
            {results && (
                <div className="upload-results">
                    {/* Summary bar */}
                    <div className="status-cards-grid" style={{ marginBottom: 20 }}>
                        <div className="card status-card">
                            <div className="status-card-icon blue"><FileText size={22} /></div>
                            <div>
                                <div className="status-card-value">{totalCount}</div>
                                <div className="status-card-label">Rows Analyzed</div>
                            </div>
                        </div>
                        <div className="card status-card">
                            <div className="status-card-icon red"><AlertTriangle size={22} /></div>
                            <div>
                                <div className="status-card-value" style={{ color: anomalyCount > 0 ? 'var(--color-critical)' : 'inherit' }}>{anomalyCount}</div>
                                <div className="status-card-label">Anomalies Detected</div>
                            </div>
                        </div>
                        <div className="card status-card">
                            <div className="status-card-icon green"><CheckCircle size={22} /></div>
                            <div>
                                <div className="status-card-value">{totalCount - anomalyCount}</div>
                                <div className="status-card-label">Normal Rows</div>
                            </div>
                        </div>
                    </div>

                    {/* Table */}
                    <div className="card results-table-wrap" style={{ padding: 0, overflow: 'hidden' }}>
                        <table className="results-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Stage</th>
                                    <th>Z-Score</th>
                                    <th>Threshold</th>
                                    <th>Anomaly</th>
                                    <th>Severity</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.predictions.map((p, i) => (
                                    <tr key={i}>
                                        <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                                        <td><strong>{p.stage}</strong></td>
                                        <td>{p.z_score != null ? Number(p.z_score).toFixed(4) : '—'}</td>
                                        <td>{p.threshold ?? '—'}</td>
                                        <td className={p.is_anomaly ? 'anomaly-yes' : 'anomaly-no'}>
                                            {p.is_anomaly ? '⚠ YES' : '✓ No'}
                                        </td>
                                        <td>
                                            {p.severity && (
                                                <span
                                                    className={`severity-badge ${p.severity.toLowerCase()}`}
                                                >
                                                    {p.severity}
                                                </span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}


