import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ReferenceLine, ResponsiveContainer, ReferenceArea
} from 'recharts';

function CustomTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '8px 12px',
            boxShadow: 'var(--shadow-card)',
            fontSize: '0.8rem',
            color: 'var(--text-primary)'
        }}>
            <p style={{ color: 'var(--text-muted)', marginBottom: 4, fontSize: '0.75rem' }}>Thời gian: {label}</p>
            {payload.map(p => (
                <p key={p.dataKey} style={{ color: p.color, fontWeight: 600 }}>
                    Giá trị: {Number(p.value).toFixed(2)}
                </p>
            ))}
        </div>
    );
}

export default function SensorChart({ data = [], threshold = 2.0, height = 180, dataKey = "score" }) {
    const color = '#1a73e8'; // Stitch blue for the line
    const fillColor = 'rgba(26, 115, 232, 0.1)';

    // Group adjacent anomaly points into areas
    const areas = [];
    let currentArea = null;
    
    data.forEach((d) => {
        if (d.isAnomaly) {
            if (!currentArea) {
                currentArea = { x1: d.t, x2: d.t };
            } else {
                currentArea.x2 = d.t;
            }
        } else {
            if (currentArea) {
                areas.push(currentArea);
                currentArea = null;
            }
        }
    });
    if (currentArea) areas.push(currentArea);

    return (
        <div className="chart-body">
            <ResponsiveContainer width="100%" height={height}>
                <LineChart data={data} margin={{ top: 20, right: 16, left: -24, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                    <XAxis
                        dataKey="t"
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={false}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={false}
                        domain={['auto', 'auto']}
                        tickFormatter={(val) => val === 0 ? '0' : val.toFixed(1)}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    
                    {areas.map((area, i) => (
                        <ReferenceArea key={i} x1={area.x1} x2={area.x2} fill="var(--color-critical)" fillOpacity={0.15} />
                    ))}

                    {threshold !== null && (
                        <ReferenceLine
                            y={threshold}
                            stroke="var(--color-critical)"
                            strokeDasharray="4 4"
                            strokeWidth={1.5}
                            label={{ value: 'NGƯỠNG CẢNH BÁO', position: 'top', fontSize: 9, fill: 'var(--color-critical)', fontWeight: 600, dy: -4 }}
                        />
                    )}
                    
                    <Line
                        type="monotone"
                        dataKey={dataKey}
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: '#fff', stroke: color, strokeWidth: 2 }}
                        isAnimationActive={false}
                        fill={fillColor}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
