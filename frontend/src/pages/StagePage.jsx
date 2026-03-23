import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { useWebSocket } from '../hooks/useWebSocket';
import StatusCard from '../components/StatusCard';
import SensorChart from '../components/SensorChart';

const STAGE_CONFIG = {
    P1: { threshold: 2.0, window: 30, name: 'Cấp nước thô', monitored: true, sensors: ['FIT 101', 'LIT 101'], actuators: ['P101 Status', 'MV 101'] },
    P2: { threshold: 2.0, window: 300, name: 'Xử lý hóa chất', monitored: true, sensors: ['AIT 201', 'AIT 202', 'FIT 201'], actuators: ['P203 Status', 'MV201'] },
    P3: { threshold: 2.0, window: 30, name: 'Siêu lọc', monitored: true, sensors: ['AIT 301', 'DPIT 301', 'FIT 301', 'LIT 301'], actuators: ['P301 Status', 'MV 301'] },
    P4: { threshold: 2.0, window: 30, name: 'Khử clo và thẩm thấu ngược', monitored: false },
    P5: { threshold: 2.0, window: 30, name: 'Thu hồi nước sạch', monitored: true, sensors: ['FIT 501', 'PIT 501', 'AIT 501'], actuators: ['P501 Status', 'MV 501'] },
    P6: { threshold: 2.0, window: 30, name: 'Làm sạch hệ thống', monitored: false },
};

const MAX_CHART_POINTS = 120;

export default function StagePage() {
    const { stageId } = useParams();
    const stageKey = stageId.toUpperCase();
    const config = STAGE_CONFIG[stageKey];

    const [chartData, setChartData] = useState({});
    const [currentData, setCurrentData] = useState({});
    const [status, setStatus] = useState({ isAnomaly: false, message: 'CHỜ DỮ LIỆU...' });
    const tickRef = useRef(0);

    // Initialize chart data structures
    useEffect(() => {
        if (!config || !config.monitored) return;
        setChartData((prev) => {
            const initialCharts = {};
            config.sensors.forEach(s => { initialCharts[s] = prev[s] || []; });
            return initialCharts;
        });
    }, [stageKey, config]);

    const handleMessage = useCallback((msg) => {
        if (!config || !config.monitored) return;
        if (msg.type === 'sensor_update') {
            const { stages, raw_data } = msg;

            const stageStatus = stages.find(s => s.stage === stageKey);
            if (stageStatus) {
                tickRef.current += 1;
                const t = tickRef.current;
                
                setCurrentData(() => {
                    const update = {};
                    config.sensors.forEach(sensor => { update[sensor] = parseFloat(raw_data[sensor] || 0); });
                    config.actuators.forEach(act => { update[act] = parseFloat(raw_data[act] || 0); });
                    return update;
                });

                setChartData(prev => {
                    const next = { ...prev };
                    config.sensors.forEach(sensor => {
                        const rawVal = parseFloat(raw_data[sensor] || 0);
                        const arr = [...(prev[sensor] || []), { t, value: rawVal, isAnomaly: stageStatus.is_anomaly }];
                        next[sensor] = arr.slice(-MAX_CHART_POINTS);
                    });
                    return next;
                });

                setStatus({
                    isAnomaly: stageStatus.is_anomaly,
                    message: stageStatus.is_anomaly ? 'CÓ BẤT THƯỜNG' : 'HOẠT ĐỘNG BÌNH THƯỜNG'
                });
            }
        }
    }, [stageKey, config]);

    useWebSocket({ onMessage: handleMessage });

    if (!config) return <Navigate to="/" />;

    if (!config.monitored) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <header className="page-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className="page-title" style={{ color: 'var(--text-muted)' }}>Tổng quan</span>
                        <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--text-muted)' }}>chevron_right</span>
                        <h1 className="page-title">{stageKey} - {config.name}</h1>
                    </div>
                </header>

                <div className="page-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', background: 'var(--bg-surface)', padding: '64px', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
                        <span className="material-symbols-outlined" style={{ fontSize: 64, marginBottom: 16, opacity: 0.5 }}>block</span>
                        <h2 style={{ fontSize: '1.4rem', marginBottom: 12, color: 'var(--text-primary)' }}>Giai đoạn này không được AI giám sát</h2>
                        <p style={{ fontSize: '1rem', lineHeight: 1.6, maxWidth: 400, margin: '0 auto' }}>Hệ thống ML (Machine Learning) không thu thập và dự đoán bất thường đối với thông số của giai đoạn <strong>{stageKey}</strong>.</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <header className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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
                        sub={`Tổng cảm biến: ${config.sensors.length}`}
                        icon="network_check"
                        iconColor={status.isAnomaly ? 'red' : 'green'}
                    />
                </div>

                <div className="card" style={{ marginBottom: 24, padding: 24 }}>
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Nhóm Cảm biến Liên tục</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24 }}>
                        {config.sensors.map(sensor => (
                            <div key={sensor} style={{ border: '1px solid var(--border-subtle)', borderRadius: 8, padding: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                                    <div style={{ fontWeight: 600 }}>{sensor}</div>
                                    <div style={{ color: status.isAnomaly ? 'var(--color-critical)' : 'var(--text-primary)', fontWeight: 700 }}>
                                        {currentData[sensor] !== undefined ? currentData[sensor].toFixed(2) : '--'}
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
                    <h3 className="card-title" style={{ marginBottom: 16 }}>Nhóm Thiết bị Chấp hành (Bơm/Van)</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
                        {config.actuators.map(act => {
                            const val = currentData[act];
                            const isActive = val && val > 0;
                            return (
                                <div key={act} style={{ 
                                    border: `1px solid ${isActive ? 'var(--color-normal)' : 'var(--border-subtle)'}`, 
                                    borderRadius: 8, padding: 16, display: 'flex', alignItems: 'center', gap: 12,
                                    background: isActive ? '#f0fdf4' : 'transparent'
                                }}>
                                    <div style={{
                                        width: 12, height: 12, borderRadius: '50%',
                                        background: isActive ? 'var(--color-normal)' : 'var(--border-card)',
                                        boxShadow: isActive ? '0 0 8px var(--color-normal)' : 'none'
                                    }} />
                                    <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{act}</div>
                                    <div style={{ marginLeft: 'auto', fontSize: '0.8rem', color: isActive ? 'var(--color-normal)' : 'var(--text-muted)' }}>
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
