import { useEffect, useMemo, useState } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
    AreaChart,
    Area,
    ReferenceLine,
    LineChart,
    Line,
    Legend,
} from 'recharts';
import { getAlerts, getHistory, getStatus } from '../services/api';
import { MONITORED_STAGES, MONITORING_ONLY_STAGES, STAGE_CONFIG, STAGES } from '../constants/stages';

const SNAPSHOT_OPTIONS = [
    { value: 'current', label: 'Current snapshot' },
    { value: 'today', label: 'Today snapshot' },
    { value: 'week', label: 'Last 7 days label' },
];

const STAGE_OPTIONS = [
    { value: 'ALL', label: 'Toan bo nha may' },
    ...STAGES.map((stage) => ({ value: stage, label: `${stage} - ${STAGE_CONFIG[stage].name}` })),
];

const STAGE_COLORS = {
    P1: '#94a3b8',
    P2: '#2563eb',
    P3: '#14b8a6',
    P4: '#f97316',
    P5: '#ef4444',
    P6: '#64748b',
};

const SEVERITY_COLORS = {
    DANGER: '#ef4444',
    CRITICAL: '#ef4444',
    HIGH: '#f97316',
    MEDIUM: '#f59e0b',
    LOW: '#3b82f6',
    NORMAL: '#10b981',
};

const EMPTY_STAGE_SUMMARY = {
    latestZ: null,
    peakZ: null,
    anomalyRows: 0,
    rowCount: 0,
    latestTimestamp: null,
};

function formatPercent(value, digits = 1) {
    return `${Number(value || 0).toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
    if (!Number.isFinite(value)) return '--';
    return Number(value).toFixed(digits);
}

function formatTimestamp(ts) {
    if (!ts) return '--';
    return new Date(ts).toLocaleString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
    });
}

function getStageHistorySummary(rows = []) {
    if (!rows.length) return EMPTY_STAGE_SUMMARY;
    const latest = rows[rows.length - 1];
    return {
        latestZ: Number.isFinite(latest?.z_score) ? latest.z_score : null,
        peakZ: rows.reduce((peak, row) => (
            Number.isFinite(row?.z_score) ? Math.max(peak, row.z_score) : peak
        ), Number.NEGATIVE_INFINITY),
        anomalyRows: rows.filter((row) => row?.is_anomaly).length,
        rowCount: rows.length,
        latestTimestamp: latest?.timestamp || null,
    };
}

function buildTrendSeries(historiesByStage, stages, maxPoints = 40) {
    const sliced = Object.fromEntries(
        stages.map((stage) => [stage, (historiesByStage[stage] || []).slice(-maxPoints)])
    );
    const maxLength = Math.max(0, ...Object.values(sliced).map((rows) => rows.length));

    return Array.from({ length: maxLength }, (_, idx) => {
        const row = { t: idx + 1 };
        stages.forEach((stage) => {
            const rows = sliced[stage];
            const point = rows[idx];
            row[stage] = Number.isFinite(point?.z_score) ? point.z_score : null;
        });
        return row;
    });
}

function buildDistribution(items, key, formatter) {
    const counts = items.reduce((acc, item) => {
        const name = item?.[key] || 'UNKNOWN';
        acc[name] = (acc[name] || 0) + 1;
        return acc;
    }, {});

    return Object.entries(counts).map(([name, value]) => ({
        name: formatter ? formatter(name) : name,
        rawName: name,
        value,
    }));
}

function ReportMetricCard({ title, value, sub, accent = 'var(--accent-primary)' }) {
    return (
        <div className="card" style={{ padding: 20 }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10 }}>
                {title}
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: accent, marginBottom: 6 }}>
                {value}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                {sub}
            </div>
        </div>
    );
}

export default function ReportsPage() {
    const [activeTab, setActiveTab] = useState('health');
    const [selectedRange, setSelectedRange] = useState('current');
    const [selectedStage, setSelectedStage] = useState('ALL');
    const [reportState, setReportState] = useState({
        loading: true,
        error: null,
        status: null,
        alerts: [],
        historiesByStage: {},
    });

    useEffect(() => {
        let cancelled = false;

        async function loadReportSnapshot() {
            setReportState((prev) => ({ ...prev, loading: true, error: null }));
            try {
                const [status, alertsResponse, stageHistories] = await Promise.all([
                    getStatus(),
                    getAlerts({ limit: 200 }),
                    Promise.all(
                        STAGES.map(async (stage) => {
                            const rows = await getHistory({ stage, limit: 240 });
                            return [stage, Array.isArray(rows) ? rows : []];
                        })
                    ),
                ]);

                if (cancelled) return;

                setReportState({
                    loading: false,
                    error: null,
                    status,
                    alerts: Array.isArray(alertsResponse) ? alertsResponse : (alertsResponse?.alerts || []),
                    historiesByStage: Object.fromEntries(stageHistories),
                });
            } catch (error) {
                if (cancelled) return;
                setReportState({
                    loading: false,
                    error: error.message || 'Failed to load report snapshot.',
                    status: null,
                    alerts: [],
                    historiesByStage: {},
                });
            }
        }

        loadReportSnapshot();
        return () => {
            cancelled = true;
        };
    }, []);

    const filteredStages = useMemo(
        () => (selectedStage === 'ALL' ? STAGES : [selectedStage]),
        [selectedStage]
    );

    const filteredMonitoredStages = useMemo(
        () => filteredStages.filter((stage) => MONITORED_STAGES.includes(stage)),
        [filteredStages]
    );

    const filteredAlerts = useMemo(
        () => (selectedStage === 'ALL'
            ? reportState.alerts
            : reportState.alerts.filter((alert) => alert.stage === selectedStage)),
        [reportState.alerts, selectedStage]
    );

    const filteredHistory = useMemo(
        () => filteredStages.flatMap((stage) => reportState.historiesByStage[stage] || []),
        [filteredStages, reportState.historiesByStage]
    );

    const stageSummaries = useMemo(
        () => Object.fromEntries(STAGES.map((stage) => [stage, getStageHistorySummary(reportState.historiesByStage[stage] || [])])),
        [reportState.historiesByStage]
    );

    const filteredStageSummaries = useMemo(
        () => filteredStages.map((stage) => ({
            stage,
            name: STAGE_CONFIG[stage].name,
            ...stageSummaries[stage],
        })),
        [filteredStages, stageSummaries]
    );

    const activeAlerts = filteredAlerts.filter((alert) => !alert.acknowledged);
    const acknowledgedAlerts = filteredAlerts.length - activeAlerts.length;
    const latestHistoryTimestamp = filteredHistory.length ? filteredHistory[filteredHistory.length - 1]?.timestamp : null;
    const currentAnomalousStages = filteredStageSummaries.filter((stage) => stage.rowCount && stage.anomalyRows > 0);
    const warmingStages = filteredStages.filter((stage) => {
        const stageStatus = reportState.status?.stages?.[stage];
        return MONITORED_STAGES.includes(stage) && stageStatus && !stageStatus.ready;
    });
    const topIncidentStage = buildDistribution(filteredAlerts, 'stage').sort((a, b) => b.value - a.value)[0];
    const securityStageDistribution = buildDistribution(filteredAlerts, 'stage').map((item) => ({
        ...item,
        fill: STAGE_COLORS[item.rawName] || '#94a3b8',
    }));
    const severityDistribution = buildDistribution(filteredAlerts, 'severity').map((item) => ({
        ...item,
        fill: SEVERITY_COLORS[item.rawName] || '#94a3b8',
    }));
    const operationsBarData = filteredStageSummaries.map((item) => ({
        stage: item.stage,
        latestZ: Number.isFinite(item.latestZ) ? Number(item.latestZ.toFixed(2)) : 0,
        peakZ: Number.isFinite(item.peakZ) ? Number(item.peakZ.toFixed(2)) : 0,
    }));
    const trendStages = selectedStage === 'ALL'
        ? MONITORED_STAGES
        : [selectedStage];
    const trendData = buildTrendSeries(reportState.historiesByStage, trendStages);
    const aiThresholdData = filteredStages
        .filter((stage) => reportState.status?.thresholds?.[stage])
        .map((stage) => ({
            stage,
            threshold: reportState.status.thresholds[stage].T,
            window: reportState.status.thresholds[stage].W,
            latestZ: stageSummaries[stage].latestZ,
            peakZ: stageSummaries[stage].peakZ,
            anomalyRate: stageSummaries[stage].rowCount
                ? (stageSummaries[stage].anomalyRows / stageSummaries[stage].rowCount) * 100
                : 0,
        }));

    const focusStage = selectedStage === 'ALL'
        ? (filteredMonitoredStages[0] || filteredStages[0] || STAGES[0])
        : selectedStage;
    const focusStageHistory = reportState.historiesByStage[focusStage] || [];
    const focusStageThreshold = reportState.status?.thresholds?.[focusStage]?.T ?? null;
    const focusStageAreaData = focusStageHistory.slice(-60).map((row, index) => ({
        t: index + 1,
        z: Number.isFinite(row?.z_score) ? row.z_score : null,
    }));

    const summaryMessage = useMemo(() => {
        const scopeLabel = selectedStage === 'ALL' ? 'the current plant snapshot' : `${selectedStage} snapshot`;
        const activeAlertLabel = activeAlerts.length === 1 ? '1 unresolved alert' : `${activeAlerts.length} unresolved alerts`;
        const warmingLabel = warmingStages.length
            ? `Warming stages: ${warmingStages.join(', ')}.`
            : 'All production stages are ready.';
        const anomalyLabel = currentAnomalousStages.length
            ? `Recent anomaly rows are present in ${currentAnomalousStages.map((stage) => stage.stage).join(', ')}.`
            : 'No anomaly rows are currently stored in the selected snapshot.';
        return `${scopeLabel} contains ${filteredHistory.length} stored history rows and ${activeAlertLabel}. ${warmingLabel} ${anomalyLabel}`;
    }, [activeAlerts.length, currentAnomalousStages, filteredHistory.length, selectedStage, warmingStages]);

    const handlePrint = () => {
        window.print();
    };

    if (reportState.loading) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '0 32px' }}>
                <div className="page-container" style={{ display: 'grid', placeItems: 'center', minHeight: 400 }}>
                    <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                        <div className="loading-ring" style={{ margin: '0 auto 16px' }} />
                        Building report snapshot from dashboard, alerts, and history...
                    </div>
                </div>
            </div>
        );
    }

    if (reportState.error) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '0 32px' }}>
                <div className="page-container">
                    <div className="card" style={{ padding: 24, color: 'var(--color-critical)' }}>
                        Failed to load report data: {reportState.error}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '0 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '24px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                <div>
                    <h1 className="page-title" style={{ fontSize: '1.5rem', marginBottom: 8 }}>Bao cao Phan tich</h1>
                    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                        <select
                            className="form-select"
                            value={selectedRange}
                            onChange={(event) => setSelectedRange(event.target.value)}
                            style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}
                        >
                            {SNAPSHOT_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                        </select>
                        <select
                            className="form-select"
                            value={selectedStage}
                            onChange={(event) => setSelectedStage(event.target.value)}
                            style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)', minWidth: 240 }}
                        >
                            {STAGE_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: 12 }}>
                    <button
                        onClick={handlePrint}
                        style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 20px', background: 'var(--accent-primary)', color: '#fff', border: 'none', borderRadius: 8, fontWeight: 600, cursor: 'pointer' }}
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: 20 }}>picture_as_pdf</span>
                        Print snapshot
                    </button>
                    <button
                        type="button"
                        disabled
                        title="Export will be added after the real-data report is finalized."
                        style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 20px', background: 'var(--bg-surface)', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)', borderRadius: 8, fontWeight: 600, cursor: 'not-allowed' }}
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: 20 }}>download</span>
                        Excel soon
                    </button>
                </div>
            </div>

            <div className="page-container" style={{ padding: '24px 0', maxWidth: 1400, margin: '0 auto', width: '100%' }}>
                <div className="card" style={{ padding: 24, marginBottom: 24 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                        <span className="material-symbols-outlined" style={{ color: 'var(--color-medium)', fontSize: 28 }}>lightbulb</span>
                        <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Operational report snapshot</h2>
                    </div>
                    <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                        {summaryMessage}
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 16, marginTop: 20 }}>
                        <ReportMetricCard
                            title="Snapshot rows"
                            value={filteredHistory.length}
                            sub={`Latest data point: ${formatTimestamp(latestHistoryTimestamp)}`}
                            accent="var(--accent-primary)"
                        />
                        <ReportMetricCard
                            title="Active alerts"
                            value={activeAlerts.length}
                            sub={`${acknowledgedAlerts} acknowledged alerts in the selected scope`}
                            accent="var(--color-critical)"
                        />
                        <ReportMetricCard
                            title="Top incident stage"
                            value={topIncidentStage?.name || '--'}
                            sub={topIncidentStage ? `${topIncidentStage.value} incidents in snapshot` : 'No incidents recorded'}
                            accent="var(--color-high)"
                        />
                        <ReportMetricCard
                            title="Monitoring-only stages"
                            value={MONITORING_ONLY_STAGES.join(', ')}
                            sub="Visible for analysis, excluded from final alert generation"
                            accent="var(--text-primary)"
                        />
                    </div>
                </div>

                <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', marginBottom: 24 }}>
                    {[
                        ['health', 'Suc khoe van hanh'],
                        ['security', 'An ninh va su co'],
                        ['ai', 'Hieu suat AI'],
                    ].map(([id, label]) => (
                        <button
                            key={id}
                            onClick={() => setActiveTab(id)}
                            style={{
                                padding: '12px 24px',
                                background: 'none',
                                border: 'none',
                                borderBottom: activeTab === id ? '3px solid var(--accent-primary)' : '3px solid transparent',
                                color: activeTab === id ? 'var(--accent-primary)' : 'var(--text-muted)',
                                fontWeight: 600,
                                fontSize: '1rem',
                                cursor: 'pointer',
                            }}
                        >
                            {label}
                        </button>
                    ))}
                </div>

                {activeTab === 'health' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 24 }}>
                        <div className="card" style={{ padding: 24 }}>
                            <h3 className="card-title" style={{ marginBottom: 20 }}>Recent Z-score trend</h3>
                            <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={trendData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                                    <XAxis dataKey="t" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                    <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                    <RechartsTooltip />
                                    <Legend />
                                    {trendStages.map((stage) => (
                                        <Line
                                            key={stage}
                                            type="monotone"
                                            dataKey={stage}
                                            name={stage}
                                            stroke={STAGE_COLORS[stage]}
                                            strokeWidth={2}
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Stage snapshot</h3>
                                <ResponsiveContainer width="100%" height={240}>
                                    <BarChart data={operationsBarData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                                        <XAxis dataKey="stage" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                        <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                        <RechartsTooltip />
                                        <Bar dataKey="latestZ" name="Latest Z-score" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
                                        <Bar dataKey="peakZ" name="Peak Z-score" fill="var(--color-medium)" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Latest stage activity</h3>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                    {filteredStageSummaries.map((item) => (
                                        <div key={item.stage} style={{ paddingBottom: 12, borderBottom: '1px solid var(--border-subtle)' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                                                <strong>{item.stage}</strong>
                                                <span style={{ color: 'var(--text-muted)' }}>{formatTimestamp(item.latestTimestamp)}</span>
                                            </div>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                                Current Z-score: <strong>{formatNumber(item.latestZ)}</strong> | Peak Z-score: <strong>{formatNumber(item.peakZ)}</strong> | Anomaly rows: <strong>{item.anomalyRows}</strong>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'security' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 24 }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 20 }}>Incident distribution by stage</h3>
                                {securityStageDistribution.length ? (
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 20 }}>
                                        <ResponsiveContainer width="55%" height={280}>
                                            <PieChart>
                                                <Pie
                                                    data={securityStageDistribution}
                                                    cx="50%"
                                                    cy="50%"
                                                    innerRadius={60}
                                                    outerRadius={100}
                                                    paddingAngle={2}
                                                    dataKey="value"
                                                >
                                                    {securityStageDistribution.map((entry) => (
                                                        <Cell key={entry.rawName} fill={entry.fill} />
                                                    ))}
                                                </Pie>
                                                <RechartsTooltip />
                                            </PieChart>
                                        </ResponsiveContainer>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minWidth: 180 }}>
                                            {securityStageDistribution.map((item) => (
                                                <div key={item.rawName} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                                    <div style={{ width: 12, height: 12, borderRadius: '50%', background: item.fill }} />
                                                    <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{item.name} ({item.value})</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div style={{ color: 'var(--text-muted)' }}>No alerts stored for the selected scope.</div>
                                )}
                            </div>

                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Recent incidents</h3>
                                <table style={{ width: '100%', fontSize: '0.85rem', textAlign: 'left', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                                            <th style={{ paddingBottom: 8 }}>Time</th>
                                            <th style={{ paddingBottom: 8 }}>Stage</th>
                                            <th style={{ paddingBottom: 8 }}>Severity</th>
                                            <th style={{ paddingBottom: 8 }}>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredAlerts.slice(0, 8).map((alert) => (
                                            <tr key={alert.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                                <td style={{ padding: '10px 0' }}>{formatTimestamp(alert.created_at)}</td>
                                                <td>{alert.stage}</td>
                                                <td style={{ color: SEVERITY_COLORS[alert.severity] || 'var(--text-primary)', fontWeight: 700 }}>{alert.severity}</td>
                                                <td>{alert.acknowledged ? 'Acknowledged' : 'Unresolved'}</td>
                                            </tr>
                                        ))}
                                        {!filteredAlerts.length && (
                                            <tr>
                                                <td colSpan={4} style={{ paddingTop: 12, color: 'var(--text-muted)' }}>
                                                    No incidents available in the current snapshot.
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <ReportMetricCard
                                title="Total incidents"
                                value={filteredAlerts.length}
                                sub={`${activeAlerts.length} unresolved, ${acknowledgedAlerts} acknowledged`}
                                accent="var(--color-critical)"
                            />
                            <ReportMetricCard
                                title="Acknowledgement rate"
                                value={filteredAlerts.length ? formatPercent((acknowledgedAlerts / filteredAlerts.length) * 100) : '0.0%'}
                                sub="Based on current stored alert records"
                                accent="var(--color-normal)"
                            />
                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Severity mix</h3>
                                {severityDistribution.length ? (
                                    <ResponsiveContainer width="100%" height={220}>
                                        <BarChart data={severityDistribution}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                                            <XAxis dataKey="name" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                            <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                            <RechartsTooltip />
                                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                                {severityDistribution.map((entry) => (
                                                    <Cell key={entry.rawName} fill={entry.fill} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div style={{ color: 'var(--text-muted)' }}>No severity data available.</div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'ai' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 24 }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <ReportMetricCard
                                title="Snapshot throughput"
                                value={filteredHistory.length}
                                sub="Stored sensor snapshots contributing to this report scope"
                                accent="var(--accent-primary)"
                            />
                            <ReportMetricCard
                                title="Coverage"
                                value={selectedStage === 'ALL' ? `${MONITORED_STAGES.length}/${STAGES.length}` : (MONITORED_STAGES.includes(selectedStage) ? 'Production' : 'Analysis only')}
                                sub={`Monitoring-only stages: ${MONITORING_ONLY_STAGES.join(', ')}`}
                                accent="var(--text-primary)"
                            />
                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Live thresholds</h3>
                                <div style={{ display: 'grid', gap: 12 }}>
                                    {aiThresholdData.map((item) => (
                                        <div key={item.stage} style={{ paddingBottom: 12, borderBottom: '1px solid var(--border-subtle)' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                                                <strong>{item.stage}</strong>
                                                <span style={{ color: 'var(--text-muted)' }}>Window {item.window}</span>
                                            </div>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                                Threshold: <strong>{formatNumber(item.threshold, 3)}</strong> | Latest Z-score: <strong>{formatNumber(item.latestZ)}</strong> | Peak Z-score: <strong>{formatNumber(item.peakZ)}</strong>
                                            </div>
                                        </div>
                                    ))}
                                    {!aiThresholdData.length && (
                                        <div style={{ color: 'var(--text-muted)' }}>No threshold data available.</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 20 }}>Focus stage Z-score vs threshold</h3>
                                <ResponsiveContainer width="100%" height={300}>
                                    <AreaChart data={focusStageAreaData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                                        <XAxis dataKey="t" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                        <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                        <RechartsTooltip />
                                        {focusStageThreshold !== null && (
                                            <ReferenceLine
                                                y={focusStageThreshold}
                                                stroke="var(--color-critical)"
                                                strokeDasharray="4 4"
                                                label={{ value: `T=${formatNumber(focusStageThreshold, 2)}`, position: 'top', fill: 'var(--color-critical)', fontSize: 12 }}
                                            />
                                        )}
                                        <Area type="monotone" dataKey="z" stroke={STAGE_COLORS[focusStage]} fill="var(--accent-glow)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                                <p style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 16 }}>
                                    Focus stage: <strong>{focusStage}</strong>. This chart uses the latest stored history rows and the live threshold from the backend status payload.
                                </p>
                            </div>

                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 20 }}>Recent anomaly frequency by stage</h3>
                                <ResponsiveContainer width="100%" height={260}>
                                    <BarChart data={aiThresholdData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                                        <XAxis dataKey="stage" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                        <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                                        <RechartsTooltip formatter={(value) => `${formatNumber(value)}%`} />
                                        <Bar dataKey="anomalyRate" name="Anomaly rate %" fill="var(--color-medium)" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
