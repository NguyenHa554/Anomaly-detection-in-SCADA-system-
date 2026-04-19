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
        message: 'ĐANG KHỞI ĐỘNG',
        detail: 'Hệ thống đang tích lũy dữ liệu',
    });
    const tickRef = useRef(0);

    const handleMessage = useCallback((msg) => {
        if (msg.type !== 'sensor_update') return;

        const rawData = msg.raw_data || {};
        const stageStatus = (msg.stages || []).find((stage) => stage.stage === stageKey);

        tickRef.current += 1;
        const t = tickRef.current;

        setCurrentData(() => {
            const next = {};
            [...(config.sensors || []), ...(config.actuators || [])].forEach((field) => {
                next[field] = parseNumericValue(rawData[field]);
            });
            return next;
        });

        setChartData((prev) => {
            const next = { ...prev };
            (config.sensors || []).forEach((sensor) => {
                const value = parseNumericValue(rawData[sensor]);
                if (value === null) return;
                const isAnomaly = stageStatus ? Boolean(stageStatus.is_anomaly) : false;
                const series = [
                    ...(prev[sensor] || []),
                    { t, value, isAnomaly },
                ];
                next[sensor] = series.slice(-MAX_CHART_POINTS);
            });
            return next;
        });

        if (!config.monitored) {
            setStatus({
                mode: 'normal',
                message: 'CHƯA GIÁM SÁT',
                detail: 'Hoạt động nhưng không có mô hình AI',
            });
            return;
        }

        if (stageStatus?.status === 'warming_up') {
            setStatus({
                mode: 'warming',
                message: 'ĐANG KHỞI ĐỘNG',
                detail: `${stageStatus.buffer_fill || 0}/${stageStatus.buffer_needed || 0} mẫu đã lưu`,
            });
            return;
        }

        const isDanger = stageStatus ? Boolean(stageStatus.is_anomaly) : false;
        setStatus({
            mode: isDanger ? 'danger' : 'normal',
            message: isDanger ? 'BẤT THƯỜNG' : 'BÌNH THƯỜNG',
            detail: isDanger ? 'Phát hiện chuỗi tín hiệu bất thường' : 'Hệ thống đang hoạt động ổn định',
        });
    }, [config, stageKey]);

    useWebSocket({ onMessage: handleMessage });

    if (!config) return <Navigate to="/" />;



    const statusColor = status.mode === 'danger' ? 'critical' : status.mode === 'warming' ? 'warning' : 'normal';
    const statusIconColor = status.mode === 'danger' ? 'red' : status.mode === 'warming' ? 'orange' : 'green';

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
                        valColor={statusColor}
                        sub={`${status.detail} | Cảm biến: ${config.sensors.length} | Thiết bị: ${config.actuators.length}`}
                        icon="network_check"
                        iconColor={statusIconColor}
                    />
                </div>

                <div className="card" style={{ marginBottom: 24, padding: 24 }}>
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Biểu đồ cảm biến liên tục</h3>
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
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Thiết bị chấp hành</h3>
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
                                            Thông số hiện tại: {value !== null && value !== undefined ? value.toFixed(2) : '--'}
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
