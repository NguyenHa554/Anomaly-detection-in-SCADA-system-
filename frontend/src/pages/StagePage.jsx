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
    const [status, setStatus] = useState({ isAnomaly: false, message: 'CHỜ DỮ LIỆU...' });
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

                const series = [...(prev[sensor] || []), { t, value, isAnomaly: stageStatus.is_anomaly }];
                next[sensor] = series.slice(-MAX_CHART_POINTS);
            });

            return next;
        });

        setStatus({
            isAnomaly: stageStatus.is_anomaly,
            message: stageStatus.is_anomaly ? 'CÓ BẤT THƯỜNG' : 'HOẠT ĐỘNG BÌNH THƯỜNG',
        });
    }, [config, stageKey]);

    useWebSocket({ onMessage: handleMessage });

    if (!config) return <Navigate to="/" />;

    if (!config.monitored) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <header className="page-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="page-title" style={{ color: 'var(--text-muted)' }}>Tổng quan</span>
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
                            Giai đoạn này không được AI giám sát
                        </h2>
                        <p style={{ fontSize: '1rem', lineHeight: 1.6, maxWidth: 560, margin: '0 auto' }}>
                            Backend mới không còn trả kết quả giám sát AI cho <strong>{stageKey}</strong>.
                            UI vẫn hiển thị đầy đủ danh sách cảm biến và thiết bị chấp hành thuộc giai đoạn này.
                        </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 24 }}>
                        <DeviceListCard
                            title="Cảm biến thuộc giai đoạn"
                            icon="sensors"
                            items={config.sensors}
                            emptyMessage="Không có cảm biến nào được cấu hình cho giai đoạn này."
                        />
                        <DeviceListCard
                            title="Thiết bị chấp hành"
                            icon="tune"
                            items={config.actuators}
                            emptyMessage="Không có thiết bị chấp hành nào được cấu hình cho giai đoạn này."
                        />
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <header className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="page-title" style={{ color: 'var(--text-muted)' }}>Tổng quan</span>
                    <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--text-muted)' }}>chevron_right</span>
                    <h1 className="page-title">{stageKey} - {config.name}</h1>
                </div>
            </header>

            <div className="page-container">
                <div style={{ marginBottom: 24 }}>
                    <StatusCard
                        label="TRẠNG THÁI GIAI ĐOẠN"
                        value={status.message}
                        valColor={status.isAnomaly ? 'critical' : 'normal'}
                        sub={`Cảm biến: ${config.sensors.length} | Thiết bị chấp hành: ${config.actuators.length}`}
                        icon="network_check"
                        iconColor={status.isAnomaly ? 'red' : 'green'}
                    />
                </div>

                <div className="card" style={{ marginBottom: 24, padding: 24 }}>
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Nhóm cảm biến liên tục</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24 }}>
                        {config.sensors.map((sensor) => (
                            <div key={sensor} style={{ border: '1px solid var(--border-subtle)', borderRadius: 8, padding: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                                    <div style={{ fontWeight: 600 }}>{sensor}</div>
                                    <div style={{ color: status.isAnomaly ? 'var(--color-critical)' : 'var(--text-primary)', fontWeight: 700 }}>
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
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Nhóm thiết bị chấp hành (Bơm/Van)</h3>
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
                                            Giá trị hiện tại: {value !== null && value !== undefined ? value.toFixed(2) : '--'}
                                        </div>
                                    </div>
                                    <div
                                        style={{
                                            marginLeft: 'auto',
                                            fontSize: '0.8rem',
                                            color: isActive ? 'var(--color-normal)' : 'var(--text-muted)',
                                        }}
                                    >
                                        {isActive ? 'HOẠT ĐỘNG' : 'TẮT'}
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
