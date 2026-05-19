import { useEffect, useMemo, useRef, useState } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ReferenceLine, ResponsiveContainer, ReferenceArea
} from 'recharts';
import {
    aggregateSeriesByTime,
    CHART_WINDOW_MS,
    resolveChartTickStepMs,
} from '../services/chartSeriesStore';

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

function formatDisplayValue(value) {
    return Number.isFinite(value) ? Number(value).toFixed(4) : '--';
}

function getSortedData(data) {
    return [...data]
        .filter((point) => point && Number.isFinite(point.ts))
        .sort((left, right) => left.ts - right.ts);
}

function getVisibleData(data, domain) {
    if (!data.length || !Array.isArray(domain)) {
        return [];
    }

    const [start, end] = domain;
    return data.filter((point) => point.ts >= start && point.ts <= end);
}

function getNumericValues(data, dataKey) {
    return data
        .map((point) => Number(point[dataKey]))
        .filter((value) => Number.isFinite(value));
}

function getLatestTimestamp(data, windowMs) {
    if (data.length) {
        return data.at(-1).ts;
    }

    return windowMs ? Date.now() : null;
}

function buildTimeDomain(data, windowMs) {
    const latestTimestamp = getLatestTimestamp(data, windowMs);
    if (latestTimestamp == null) {
        return ['auto', 'auto'];
    }

    if (windowMs) {
        return [latestTimestamp - windowMs, latestTimestamp];
    }

    if (data.length > 1) {
        return [data[0].ts, data.at(-1).ts];
    }

    return [latestTimestamp - 1000, latestTimestamp + 1000];
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
    const minVisibleRange = Math.max(Math.abs(center) * 0.012, 0.05);
    const effectiveRange = Math.max(range, minVisibleRange);
    const padding = effectiveRange * 0.32;

    if (range === 0) {
        return [center - effectiveRange, center + effectiveRange];
    }

    return [min - padding, max + padding];
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

    const shrinkRate = 0.06;
    return [
        previousMin + (targetMin - previousMin) * shrinkRate,
        previousMax + (targetMax - previousMax) * shrinkRate,
    ];
}

const STICKY_Y_DOMAINS = new Map();

function getStickyYDomain(values, resetKey) {
    const targetDomain = getTargetYDomain(values);
    const previousDomain = STICKY_Y_DOMAINS.get(resetKey);
    const domain = getNextStickyDomain(previousDomain, targetDomain);

    STICKY_Y_DOMAINS.set(resetKey, domain);
    return domain;
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

    if (!ticks.length) {
        return [start, end];
    }

    if (ticks[0] - start > tickStepMs * 0.6) {
        ticks.unshift(start);
    }
    if (end - ticks.at(-1) > tickStepMs * 0.6) {
        ticks.push(end);
    }

    return ticks;
}

function useElementWidth() {
    const ref = useRef(null);
    const [width, setWidth] = useState(0);

    useEffect(() => {
        const element = ref.current;
        if (!element) {
            return undefined;
        }

        const updateWidth = () => {
            setWidth(element.getBoundingClientRect().width);
        };

        updateWidth();
        const resizeObserver = new ResizeObserver(updateWidth);
        resizeObserver.observe(element);

        return () => resizeObserver.disconnect();
    }, []);

    return [ref, width];
}

function CustomTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;

    return (
        <div className="chart-tooltip">
            <p className="chart-tooltip-time">
                Time: {formatChartTime(label, TOOLTIP_TIME_FORMATTER)}
            </p>
            {payload
                .filter((point) => Number.isFinite(Number(point.value)))
                .map((point) => (
                    <p key={point.dataKey} className="chart-tooltip-value" style={{ color: point.color }}>
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
    aggregation = 'latest',
    aggregationIntervalMs = null,
    minTickPx = 86,
}) {
    const color = '#1a73e8';
    const [containerRef, containerWidth] = useElementWidth();
    const sortedData = useMemo(() => getSortedData(data), [data]);
    const xDomain = useMemo(() => buildTimeDomain(sortedData, windowMs), [sortedData, windowMs]);
    const effectiveTickStepMs = useMemo(() => resolveChartTickStepMs({
        windowMs: Array.isArray(xDomain) ? xDomain[1] - xDomain[0] : windowMs,
        width: containerWidth,
        preferredStepMs: tickStepMs,
        minTickPx,
    }), [containerWidth, minTickPx, tickStepMs, windowMs, xDomain]);

    const visibleData = useMemo(() => getVisibleData(sortedData, xDomain), [sortedData, xDomain]);
    const effectiveAggregationIntervalMs = useMemo(() => {
        if (Number.isFinite(aggregationIntervalMs)) {
            return Math.max(1000, aggregationIntervalMs);
        }

        return Math.max(1000, Math.min(effectiveTickStepMs, 5000));
    }, [aggregationIntervalMs, effectiveTickStepMs]);
    const aggregatedData = useMemo(() => aggregateSeriesByTime(visibleData, {
        dataKey,
        intervalMs: effectiveAggregationIntervalMs,
        mode: aggregation,
    }), [aggregation, dataKey, effectiveAggregationIntervalMs, visibleData]);
    const chartData = useMemo(
        () => insertGapBreaks(aggregatedData, dataKey, gapThresholdMs),
        [aggregatedData, dataKey, gapThresholdMs]
    );
    const values = useMemo(() => getNumericValues(aggregatedData, dataKey), [aggregatedData, dataKey]);
    const domainValues = useMemo(() => (
        threshold !== null && Number.isFinite(Number(threshold))
            ? [...values, Number(threshold)]
            : values
    ), [threshold, values]);
    const yDomain = useMemo(
        () => getStickyYDomain(domainValues, resetKey),
        [domainValues, resetKey]
    );
    const yRange = values.length ? Math.max(...values) - Math.min(...values) : 1;
    const chartLatestValue = latestValue ?? visibleData.at(-1)?.[dataKey];
    const xTicks = Array.isArray(xDomain) ? buildTimeTicks(xDomain, effectiveTickStepMs) : undefined;

    const anomalyAreas = [];
    let currentArea = null;

    chartData.forEach((point) => {
        if (point.isGap) {
            if (currentArea) {
                anomalyAreas.push(currentArea);
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
            anomalyAreas.push(currentArea);
            currentArea = null;
        }
    });

    if (currentArea) {
        anomalyAreas.push(currentArea);
    }

    return (
        <div className="chart-body" ref={containerRef}>
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
                        interval={0}
                        minTickGap={18}
                        tickFormatter={(value) => formatChartTime(value, TICK_TIME_FORMATTER)}
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                        tickLine={false}
                        axisLine={{ stroke: 'var(--border-subtle)' }}
                        domain={yDomain}
                        width={58}
                        tickCount={5}
                        minTickGap={8}
                        allowDecimals
                        tickFormatter={(value) => formatYAxisTick(value, yRange)}
                    />
                    <Tooltip content={<CustomTooltip />} />

                    {anomalyAreas.map((area) => (
                        <ReferenceArea
                            key={`${area.x1}-${area.x2}`}
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
