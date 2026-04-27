import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ReferenceLine, ResponsiveContainer, ReferenceArea
} from 'recharts';

const TICK_TIME_FORMATTER = new Intl.DateTimeFormat('vi-VN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
});

const TOOLTIP_TIME_FORMATTER = new Intl.DateTimeFormat('vi-VN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
});

function formatChartTime(value, formatter) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '--' : formatter.format(date);
}

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
            <p style={{ color: 'var(--text-muted)', marginBottom: 4, fontSize: '0.75rem' }}>
                Time: {formatChartTime(label, TOOLTIP_TIME_FORMATTER)}
            </p>
            {payload.map((point) => (
                <p key={point.dataKey} style={{ color: point.color, fontWeight: 600 }}>
                    Value: {Number(point.value).toFixed(2)}
                </p>
            ))}
        </div>
    );
}

export default function SensorChart({ data = [], threshold = 2.0, height = 180, dataKey = 'score' }) {
    const color = '#1a73e8';
    const fillColor = 'rgba(26, 115, 232, 0.1)';

    const areas = [];
    let currentArea = null;

    data.forEach((point) => {
        if (point.isAnomaly) {
            if (!currentArea) {
                currentArea = { x1: point.ts, x2: point.ts };
            } else {
                currentArea.x2 = point.ts;
            }
        } else if (currentArea) {
            areas.push(currentArea);
            currentArea = null;
        }
    });

    if (currentArea) {
        areas.push(currentArea);
    }

    const xDomain = data.length > 1
        ? ['dataMin', 'dataMax']
        : data.length === 1
            ? [data[0].ts - 1000, data[0].ts + 1000]
            : ['auto', 'auto'];

    return (
        <div className="chart-body">
            <ResponsiveContainer width="100%" height={height}>
                <LineChart data={data} margin={{ top: 20, right: 16, left: -24, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                    <XAxis
                        dataKey="ts"
                        type="number"
                        scale="time"
                        domain={xDomain}
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={false}
                        interval="preserveStartEnd"
                        tickFormatter={(value) => formatChartTime(value, TICK_TIME_FORMATTER)}
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={false}
                        domain={['auto', 'auto']}
                        tickFormatter={(value) => value === 0 ? '0' : value.toFixed(1)}
                    />
                    <Tooltip content={<CustomTooltip />} />

                    {areas.map((area, index) => (
                        <ReferenceArea
                            key={index}
                            x1={area.x1}
                            x2={area.x2}
                            fill="var(--color-critical)"
                            fillOpacity={0.15}
                        />
                    ))}

                    {threshold !== null && (
                        <ReferenceLine
                            y={threshold}
                            stroke="var(--color-critical)"
                            strokeDasharray="4 4"
                            strokeWidth={1.5}
                            label={{
                                value: 'ALERT THRESHOLD',
                                position: 'top',
                                fontSize: 9,
                                fill: 'var(--color-critical)',
                                fontWeight: 600,
                                dy: -4,
                            }}
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
