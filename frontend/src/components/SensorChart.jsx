import { useMemo, useState } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ReferenceLine, ResponsiveContainer, ReferenceArea
} from 'recharts';
import { CHART_WINDOW_MS } from '../services/chartSeriesStore';

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

function getNumericValues(data, dataKey) {
    return data
        .map((point) => Number(point[dataKey]))
        .filter((value) => Number.isFinite(value));
}

function getVisibleData(data, windowMs) {
    if (!data.length) {
        return [];
    }

    const latestTimestamp = data.at(-1)?.ts;
    if (!Number.isFinite(latestTimestamp) || !windowMs) {
        return data;
    }

    const windowStart = latestTimestamp - windowMs;
    return data.filter((point) => point.ts >= windowStart && point.ts <= latestTimestamp);
}

function insertGapBreaks(data, dataKey, gapThresholdMs) {
    if (!gapThresholdMs || data.length <= 1) {
        return data;
    }

    const next = [];
    data.forEach((point, index) => {
        const previousPoint = data[index - 1];
        if (previousPoint && point.ts - previousPoint.ts > gapThresholdMs) {
            next.push({
                ...previousPoint,
                ts: previousPoint.ts + 1,
                [dataKey]: null,
                isAnomaly: false,
                isGap: true,
            });
            next.push({
                ...point,
                ts: point.ts - 1,
                [dataKey]: null,
                isAnomaly: false,
                isGap: true,
            });
        }
        next.push(point);
    });

    return next;
}

function getTargetYDomain(values) {
    if (values.length === 0) {
        return ['auto', 'auto'];
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const center = (min + max) / 2;
    const minVisibleRange = Math.max(Math.abs(center) * 0.01, 0.05);
    const effectiveRange = Math.max(range, minVisibleRange);
    const padding = effectiveRange * 0.35;

    if (range === 0) {
        return [center - effectiveRange, center + effectiveRange];
    }

    return [min - padding, max + padding];
}

function useStickyYDomain(values, resetKey) {
    const targetDomain = useMemo(() => {
        if (values.length === 0) {
            return ['auto', 'auto'];
        }

        return getTargetYDomain(values);
    }, [values]);
    const targetKey = `${resetKey}:${targetDomain.join(':')}`;
    const [stickyDomain, setStickyDomain] = useState({ key: targetKey, resetKey, domain: targetDomain });

    if (stickyDomain.key !== targetKey) {
        const nextDomain = stickyDomain.resetKey === resetKey
            ? getNextStickyDomain(stickyDomain.domain, targetDomain)
            : targetDomain;
        setStickyDomain({ key: targetKey, resetKey, domain: nextDomain });
        return nextDomain;
    }

    return stickyDomain.domain;
}

function getNextStickyDomain(previousDomain, targetDomain) {
    if (targetDomain[0] === 'auto' || !previousDomain || previousDomain[0] === 'auto') {
        return targetDomain;
    }

    const [targetMin, targetMax] = targetDomain;
    const [previousMin, previousMax] = previousDomain;
    const targetRange = targetMax - targetMin;
    const previousRange = previousMax - previousMin;
    const shouldExpand = targetMin < previousMin || targetMax > previousMax;

    if (shouldExpand || targetRange > previousRange) {
        return [
            Math.min(targetMin, previousMin),
            Math.max(targetMax, previousMax),
        ];
    }

    const shrinkRate = 0.08;
    return [
        previousMin + (targetMin - previousMin) * shrinkRate,
        previousMax + (targetMax - previousMax) * shrinkRate,
    ];
}

function formatYAxisTick(value, valueRange) {
    if (value === 0) {
        return '0';
    }

    if (valueRange < 0.1) {
        return value.toFixed(3);
    }

    if (valueRange < 1) {
        return value.toFixed(2);
    }

    return value.toFixed(1);
}

function buildTimeTicks(domain, tickStepMs) {
    const [start, end] = domain;
    if (!Number.isFinite(start) || !Number.isFinite(end) || !tickStepMs) {
        return undefined;
    }

    const firstTick = Math.ceil(start / tickStepMs) * tickStepMs;
    const ticks = [];
    for (let tick = firstTick; tick <= end; tick += tickStepMs) {
        ticks.push(tick);
    }

    if (!ticks.includes(start)) {
        ticks.unshift(start);
    }
    if (!ticks.includes(end)) {
        ticks.push(end);
    }

    return ticks;
}

function formatDisplayValue(value) {
    return Number.isFinite(value) ? Number(value).toFixed(4) : '--';
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
                    Value: {Number(point.value).toFixed(4)}
                </p>
            ))}
        </div>
    );
}

export default function SensorChart({
    data = [],
    threshold = 2.0,
    height = 180,
    dataKey = 'score',
    windowMs = CHART_WINDOW_MS,
    showMiniOverview = false,
    tickStepMs = 5000,
    legendLabel = null,
    latestValue = null,
    resetKey = dataKey,
    gapThresholdMs = null,
}) {
    const color = '#1a73e8';
    const fillColor = 'rgba(26, 115, 232, 0.1)';
    const sortedData = useMemo(
        () => [...data].filter((point) => Number.isFinite(point?.ts)).sort((left, right) => left.ts - right.ts),
        [data]
    );
    const latestTimestamp = sortedData.length ? sortedData.at(-1).ts : null;
    const visibleData = useMemo(() => getVisibleData(sortedData, windowMs), [sortedData, windowMs]);
    const chartData = useMemo(
        () => insertGapBreaks(visibleData, dataKey, gapThresholdMs),
        [dataKey, gapThresholdMs, visibleData]
    );
    const values = useMemo(() => getNumericValues(visibleData, dataKey), [visibleData, dataKey]);
    const yDomain = useStickyYDomain(values, resetKey);
    const yRange = values.length ? Math.max(...values) - Math.min(...values) : 1;
    const chartLatestValue = latestValue ?? visibleData.at(-1)?.[dataKey];

    const areas = [];
    let currentArea = null;

    chartData.forEach((point) => {
        if (point.isGap) {
            if (currentArea) {
                areas.push(currentArea);
                currentArea = null;
            }
            return;
        }

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

    const xDomain = windowMs && latestTimestamp != null
        ? [latestTimestamp - windowMs, latestTimestamp]
        : sortedData.length > 1
            ? ['dataMin', 'dataMax']
        : sortedData.length === 1
            ? [sortedData[0].ts - 1000, sortedData[0].ts + 1000]
            : ['auto', 'auto'];
    const xTicks = Array.isArray(xDomain) ? buildTimeTicks(xDomain, tickStepMs) : undefined;

    return (
        <div className="chart-body">
            {legendLabel && (
                <div className="chart-meta-row">
                    <div className="chart-legend">
                        <span className="chart-legend-dot" style={{ background: color }} />
                        {legendLabel}
                    </div>
                    <div className="chart-latest">
                        <span>Latest</span>
                        <strong>{formatDisplayValue(Number(chartLatestValue))}</strong>
                    </div>
                </div>
            )}
            <ResponsiveContainer width="100%" height={height}>
                <LineChart data={chartData} margin={{ top: 16, right: 16, left: 8, bottom: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical stroke="var(--border-subtle)" />
                    <XAxis
                        dataKey="ts"
                        type="number"
                        scale="time"
                        domain={xDomain}
                        ticks={xTicks}
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={{ stroke: 'var(--border-subtle)' }}
                        interval="preserveStart"
                        minTickGap={28}
                        tickFormatter={(value) => formatChartTime(value, TICK_TIME_FORMATTER)}
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={{ stroke: 'var(--border-subtle)' }}
                        domain={yDomain}
                        width={56}
                        tickCount={4}
                        minTickGap={8}
                        allowDecimals
                        tickFormatter={(value) => formatYAxisTick(value, yRange)}
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
                        connectNulls={false}
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: '#fff', stroke: color, strokeWidth: 2 }}
                        isAnimationActive={false}
                        fill={fillColor}
                    />
                </LineChart>
            </ResponsiveContainer>
            {showMiniOverview && sortedData.length > 1 && (
                <div className="chart-mini-overview">
                    <ResponsiveContainer width="100%" height={34}>
                        <LineChart
                            data={insertGapBreaks(sortedData, dataKey, gapThresholdMs)}
                            margin={{ top: 6, right: 4, left: 4, bottom: 2 }}
                        >
                            <XAxis dataKey="ts" type="number" scale="time" hide domain={['dataMin', 'dataMax']} />
                            <YAxis hide domain={['auto', 'auto']} />
                            <Line
                                type="monotone"
                                dataKey={dataKey}
                                connectNulls={false}
                                stroke={color}
                                strokeWidth={1.5}
                                dot={false}
                                isAnimationActive={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
}
