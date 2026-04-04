import { useState, useRef, useCallback } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { useWebSocket } from '../hooks/useWebSocket';
import StatusCard from '../components/StatusCard';
import SensorChart from '../components/SensorChart';
import { STAGE_CONFIG } from '../constants/stages';

const MAX_CHART_POINTS = 120;

function parseNumericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function DeviceListCard({ title, icon, items, emptyMessage }) {
    return (
        <div className="card" style={{ padding: 24 }}>
            <div className="card-header" style={{ marginBottom: 16 }}>
                <h3 className="card-title">
                    <span className="material-symbols-outlined" style={{ color: 'var(--accent-primary)' }}>{icon}</span>
                    {title}
                </h3>
            </div>

            {items.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                    {items.map((item) => (
                        <div
                            key={item}
                            style={{
                                padding: '12px 14px',
                                borderRadius: 10,
                                border: '1px solid var(--border-subtle)',
                                background: 'var(--bg-surface)',
                                fontWeight: 600,
                            }}
                        >
                            {item}
                        </div>
                    ))}
                </div>
            ) : (
                <div style={{ color: 'var(--text-muted)' }}>{emptyMessage}</div>
            )}
        </div>
    );
}

export default function StagePage() {
    const { stageId } = useParams();
    const stageKey = stageId?.toUpperCase();
    const config = STAGE_CONFIG[stageKey];

    const [chartData, setChartData] = useState({});
    const [currentData, setCurrentData] = useState({});
    const [status, setStatus] = useState({
        mode: 'warming',
        message: 'WARMING UP',
        detail: 'Waiting for enough samples',
    });
    const tickRef = useRef(0);

    const handleMessage = useCallback((msg) => {
        if (!config?.monitored || msg.type !== 'sensor_update') return;

        const rawData = msg.raw_data || {};
        const stageStatus = (msg.stages || []).find((stage) => stage.stage === stageKey);
        if (!stageStatus) return;

        tickRef.current += 1;
        const t = tickRef.current;

        setCurrentData(() => {
            const next = {};
            [...config.sensors, ...config.actuators].forEach((field) => {
                next[field] = parseNumericValue(rawData[field]);
            });
            return next;
        });

        setChartData((prev) => {
            const next = { ...prev };
            config.sensors.forEach((sensor) => {
                const value = parseNumericValue(rawData[sensor]);
                if (value === null) return;
                const series = [
                    ...(prev[sensor] || []),
                    { t, value, isAnomaly: Boolean(stageStatus.is_anomaly) },
                ];
                next[sensor] = series.slice(-MAX_CHART_POINTS);
            });
            return next;
        });

        if (stageStatus.status === 'warming_up') {
            setStatus({
                mode: 'warming',
                message: 'WARMING UP',
                detail: `${stageStatus.buffer_fill || 0}/${stageStatus.buffer_needed || 0} samples ready`,
            });
            return;
        }

        const isDanger = Boolean(stageStatus.is_anomaly);
        setStatus({
            mode: isDanger ? 'danger' : 'normal',
            message: isDanger ? 'DANGER' : 'NORMAL',
            detail: isDanger ? 'Confirmed anomaly episode detected' : 'Detector ready and normal',
        });
    }, [config, stageKey]);

    useWebSocket({ onMessage: handleMessage });

    if (!config) return <Navigate to="/" />;

    if (!config.monitored) {
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
                    <div
                        className="card"
                        style={{
                            marginBottom: 24,
                            padding: 32,
                            textAlign: 'center',
                            color: 'var(--text-muted)',
                        }}
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: 64, marginBottom: 16, opacity: 0.5 }}>block</span>
                        <h2 style={{ fontSize: '1.4rem', marginBottom: 12, color: 'var(--text-primary)' }}>
                            This stage is not monitored by AI
                        </h2>
                        <p style={{ fontSize: '1rem', lineHeight: 1.6, maxWidth: 560, margin: '0 auto' }}>
                            The backend does not return AI monitoring results for <strong>{stageKey}</strong>.
                            The UI still shows the sensors and actuators assigned to this stage.
                        </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 24 }}>
                        <DeviceListCard
                            title="Stage sensors"
                            icon="sensors"
                            items={config.sensors}
                            emptyMessage="No sensors configured for this stage."
                        />
                        <DeviceListCard
                            title="Actuators"
                            icon="tune"
                            items={config.actuators}
                            emptyMessage="No actuators configured for this stage."
                        />
                    </div>
                </div>
            </div>
        );
    }

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
                            <div key={sensor} style={{ border: '1px solid var(--border-subtle)', borderRadius: 8, padding: 16 }}>
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
                                    height={150}
                                    dataKey="value"
                                />
                            </div>
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
        </div>
    );
}
