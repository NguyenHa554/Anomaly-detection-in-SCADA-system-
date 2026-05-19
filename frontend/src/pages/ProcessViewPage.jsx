import { useCallback, useEffect, useRef, useState } from 'react';
import { getHistory } from '../services/api';
import { useScadaStream } from '../context/scadaStreamContextValue';
import DeviceHistoryModal from '../components/DeviceHistoryModal';
import {
    buildDeviceHistory,
    DAY_HISTORY_LIMIT,
} from '../services/deviceHistory';

const PROCESS_CANVAS = { width: 1720, height: 920 };

const PROCESS_GROUPS = [
    { id: 'P1', label: 'P1', x: 40, y: 82, width: 440, height: 260 },
    { id: 'P2', label: 'P2', x: 508, y: 48, width: 620, height: 292 },
    { id: 'P3', label: 'P3', x: 1180, y: 82, width: 490, height: 366, labelAlign: 'left' },
    { id: 'P4', label: 'P4', x: 40, y: 430, width: 770, height: 240, labelAlign: 'left' },
    { id: 'P5', label: 'P5', x: 40, y: 702, width: 820, height: 160 },
    { id: 'P6', label: 'P6', x: 980, y: 610, width: 620, height: 230 },
];

const PROCESS_EQUIPMENT = [
    { id: 'raw-tank', stage: 'P1', type: 'largeTank', label: 'Raw Water Tank', tag: 'LIT 101', x: 80, y: 120, width: 176, height: 206 },
    { id: 'p101', stage: 'P1', type: 'pump', label: 'P101 Pump', tag: 'P101 Status', x: 332, y: 184, width: 126, height: 92 },

    { id: 'hcl', stage: 'P2', type: 'chemicalTank', label: 'HCL', x: 548, y: 76, width: 104, height: 74 },
    { id: 'naocl', stage: 'P2', type: 'chemicalTank', label: 'NaOCl', x: 710, y: 76, width: 112, height: 74 },
    { id: 'nacl', stage: 'P2', type: 'chemicalTank', label: 'NaCl', x: 878, y: 76, width: 104, height: 74 },
    { id: 'p201', stage: 'P2', type: 'dosingPump', label: 'P201', tag: 'P201 Status', x: 574, y: 176, width: 82, height: 66 },
    { id: 'p203', stage: 'P2', type: 'dosingPump', label: 'P203', tag: 'P203 Status', x: 740, y: 176, width: 82, height: 66 },
    { id: 'p205', stage: 'P2', type: 'dosingPump', label: 'P205', tag: 'P205 Status', x: 906, y: 176, width: 82, height: 66 },
    { id: 'static-mixer', stage: 'P2', type: 'mixer', label: 'Static Mixer', x: 1020, y: 196, width: 92, height: 82 },

    { id: 'uf-feed-tank', stage: 'P3', type: 'largeTank', label: 'UF Feed Tank', tag: 'LIT 301', x: 1462, y: 122, width: 176, height: 206 },
    { id: 'p301', stage: 'P3', type: 'pump', label: 'P301 UF Feed Pump', tag: 'P301 Status', x: 1460, y: 348, width: 138, height: 96 },
    { id: 'uf-unit', stage: 'P3', type: 'filterUnit', label: 'Ultrafiltration Unit (UF)', x: 1208, y: 362, width: 190, height: 88 },

    { id: 'uv-dechlor', stage: 'P4', type: 'uv', label: 'Ultraviolet Dechlorination', tag: 'UV401', x: 78, y: 488, width: 184, height: 92 },
    { id: 'nahso3', stage: 'P4', type: 'chemicalTank', label: 'NaHSO3', x: 300, y: 578, width: 122, height: 70 },
    { id: 'p401', stage: 'P4', type: 'pump', label: 'P401 RO Feed Pump', tag: 'P401 Status', x: 430, y: 492, width: 138, height: 96 },
    { id: 'ro-feed-tank', stage: 'P4', type: 'largeTank', label: 'RO Feed Tank', tag: 'LIT 401', x: 616, y: 452, width: 176, height: 206 },

    { id: 'cartridge-filter', stage: 'P5', type: 'filterUnit', label: 'Cartridge Filter', x: 78, y: 744, width: 150, height: 88 },
    { id: 'p501', stage: 'P5', type: 'pump', label: 'P501 RO Boost Pump', tag: 'P501 Status', x: 316, y: 740, width: 138, height: 96 },
    { id: 'ro-unit', stage: 'P5', type: 'roUnit', label: 'Reverse Osmosis Unit', x: 544, y: 742, width: 242, height: 92 },

    { id: 'uf-backwash-tank', stage: 'P6', type: 'staticTank', label: 'UF Backwash Tank', x: 1016, y: 642, width: 174, height: 82 },
    { id: 'p602', stage: 'P6', type: 'pump', label: 'P602 Backwash Pump', tag: 'P602 Status', x: 1288, y: 636, width: 138, height: 96 },
    { id: 'raw-permeate-tank', stage: 'P6', type: 'staticTank', label: 'Raw Permeate Tank', x: 1016, y: 752, width: 174, height: 82 },
    { id: 'water-recycled', stage: 'P6', type: 'outlet', label: 'Water recycled', x: 1290, y: 770, width: 178, height: 62 },
];

const PROCESS_SENSORS = [
    { stage: 'P2', tag: 'FIT 201', label: 'FIT201', x: 510, y: 284 },
    { stage: 'P2', tag: 'AIT 201', label: 'AIT201', x: 594, y: 284 },
    { stage: 'P2', tag: 'AIT 202', label: 'AIT202', x: 1118, y: 286 },
    { stage: 'P2', tag: 'AIT 203', label: 'AIT203', x: 1210, y: 286 },
    { stage: 'P3', tag: 'DPIT 301', label: 'DPIT301', x: 1258, y: 320 },
    { stage: 'P4', tag: 'FIT 401', label: 'FIT401', x: 278, y: 486 },
    { stage: 'P4', tag: 'AIT 402', label: 'AIT402', x: 78, y: 598 },
    { stage: 'P5', tag: 'AIT 503', label: 'AIT503', x: 232, y: 806 },
    { stage: 'P5', tag: 'AIT 504', label: 'AIT504', x: 700, y: 840 },
];

const PROCESS_VALVES = [
    { id: 'mv101', stage: 'P1', tag: 'MV 101', x: 300, y: 230 },
    { id: 'mv201', stage: 'P2', tag: 'MV201', x: 522, y: 230 },
    { id: 'chem-main', x: 992, y: 230 },
    { id: 'p3-return', x: 1430, y: 396 },
    { id: 'uv-inlet', x: 308, y: 540 },
    { id: 'p4-tank-outlet', x: 592, y: 540 },
    { id: 'p5-filter', x: 274, y: 788 },
    { id: 'ro-permeate', x: 856, y: 788 },
    { id: 'ro-reject', x: 858, y: 682 },
];

const PROCESS_PIPES = [
    { id: 'p1-main-left', d: 'M256 230 H272', kind: 'main' },
    { id: 'p1-main-right', d: 'M328 230 H336', kind: 'main' },
    { id: 'p1-p2-main-left', d: 'M458 230 H494', kind: 'main' },
    { id: 'p1-p2-main-right', d: 'M550 230 H574', kind: 'main' },
    { id: 'p2-main-p201-p203', d: 'M656 230 H740', kind: 'main' },
    { id: 'p2-main-p203-p205', d: 'M822 230 H906', kind: 'main' },
    { id: 'p2-main-p205-valve', d: 'M988 230 H964', kind: 'main' },
    { id: 'hcl-drop', d: 'M600 150 V176', kind: 'chemical' },
    { id: 'naocl-drop', d: 'M766 150 V176', kind: 'chemical' },
    { id: 'nacl-drop', d: 'M932 150 V176', kind: 'chemical' },
    { id: 'p201-injection', d: 'M615 242 V230', kind: 'chemical' },
    { id: 'p203-injection', d: 'M781 242 V230', kind: 'chemical' },
    { id: 'p205-injection', d: 'M947 242 V230', kind: 'chemical' },
    { id: 'p2-p3-feed', d: 'M1112 238 H1550 V122', kind: 'main' },
    { id: 'tank-to-p301', d: 'M1550 328 V348', kind: 'main' },
    { id: 'p301-to-valve', d: 'M1464 396 H1458', kind: 'main' },
    { id: 'valve-to-uf', d: 'M1402 396 H1398', kind: 'main' },
    { id: 'uf-to-p4', d: 'M1208 404 H792 V540 H620', kind: 'main' },
    { id: 'p4-tank-valve-to-p401', d: 'M564 540 H568', kind: 'main' },
    { id: 'p401-to-uv-left', d: 'M434 540 H336', kind: 'main', reverse: true },
    { id: 'p401-to-uv-right', d: 'M280 540 H262', kind: 'main', reverse: true },
    { id: 'uv-to-p5-down', d: 'M170 580 V744', kind: 'main' },
    { id: 'p5-main-1-left', d: 'M228 788 H246', kind: 'main' },
    { id: 'p5-main-1-right', d: 'M302 788 H320', kind: 'main' },
    { id: 'p5-main-2', d: 'M450 788 H544', kind: 'main' },
    { id: 'ro-to-permeate-left', d: 'M786 788 H828', kind: 'main' },
    { id: 'ro-to-permeate-right', d: 'M884 788 H1016', kind: 'main' },
    { id: 'ro-to-reject-drop', d: 'M786 788 V682 H830', kind: 'return' },
    { id: 'ro-to-reject-right', d: 'M886 682 H1016', kind: 'return' },
    { id: 'backwash-to-p602', d: 'M1190 684 H1292', kind: 'main' },
    { id: 'raw-permeate-out', d: 'M1190 792 H1290', kind: 'main' },
    { id: 'p602-recycle', d: 'M1422 684 C1510 560 1522 476 1398 404', kind: 'return' },
    { id: 'uf-backwash-return', d: 'M1208 386 C1110 350 994 424 792 524', kind: 'return' },
];

const FLOW_LABELS = [
    { text: 'Raw water feed', x: 270, y: 204 },
    { text: 'Chemical dosing station', x: 676, y: 266 },
    { text: 'UF filtrate / RO feed', x: 842, y: 516 },
    { text: 'P: Permeate', x: 876, y: 774 },
    { text: 'R: Reject', x: 874, y: 668 },
];

const FLOW_ARROWS = [
    { id: 'raw-to-p101', x: 306, y: 230, rotate: 0 },
    { id: 'p101-to-chem', x: 530, y: 230, rotate: 0 },
    { id: 'chem-to-mixer', x: 1002, y: 230, rotate: 0 },
    { id: 'mixer-to-uf-tank', x: 1348, y: 238, rotate: 0 },
    { id: 'uf-tank-to-p301', x: 1550, y: 338, rotate: 90 },
    { id: 'p301-to-uf', x: 1428, y: 396, rotate: 180 },
    { id: 'uf-to-ro-feed', x: 1038, y: 540, rotate: 180 },
    { id: 'ro-feed-to-p401', x: 594, y: 540, rotate: 180 },
    { id: 'p401-to-uv', x: 328, y: 540, rotate: 180 },
    { id: 'uv-to-p5', x: 170, y: 700, rotate: 90 },
    { id: 'filter-to-p501', x: 274, y: 788, rotate: 0 },
    { id: 'p501-to-ro', x: 502, y: 788, rotate: 0 },
    { id: 'ro-to-permeate', x: 946, y: 788, rotate: 0 },
    { id: 'ro-to-reject', x: 944, y: 682, rotate: 0 },
    { id: 'backwash-to-p602', x: 1240, y: 684, rotate: 0 },
    { id: 'permeate-to-recycle', x: 1240, y: 792, rotate: 0 },
];

const TANK_LEVEL_RANGES = {
    'LIT 101': { min: 0, max: 1000 },
    'LIT 301': { min: 0, max: 1200 },
    'LIT 401': { min: 0, max: 1200 },
};

function parseNumericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function formatValue(value) {
    const parsed = parseNumericValue(value);
    return parsed == null ? '--' : parsed.toFixed(2);
}

function valueFor(stageCurrentData, stage, tag) {
    return stageCurrentData?.[stage]?.[tag];
}

function isActive(value) {
    const parsed = parseNumericValue(value);
    return parsed != null && parsed > 0;
}

function tankLevelPercent(tag, value) {
    if (!tag) return 0.46;
    const parsed = parseNumericValue(value);
    const range = TANK_LEVEL_RANGES[tag];
    if (parsed == null || !range) return 0.46;
    const ratio = (parsed - range.min) / (range.max - range.min);
    return Math.min(0.95, Math.max(0.05, ratio));
}

function levelPercentLabel(tag, value) {
    return `${Math.round(tankLevelPercent(tag, value) * 100)}%`;
}

function openableDevice(stage, field, kind = 'sensor') {
    return { stage, field, kind };
}

function LargeTankSymbol({ id, tag, value }) {
    const level = tankLevelPercent(tag, value);
    const fillTop = 38;
    const fillBottom = 188;
    const fillRange = fillBottom - fillTop;
    const fillHeight = fillRange * level;
    const fillY = fillBottom - fillHeight;
    const clipId = `large-tank-fill-${id}`;
    const ticks = [0, 25, 50, 75, 100];

    return (
        <svg className="process-large-tank-symbol" viewBox="0 0 170 210" aria-hidden="true">
            <defs>
                <linearGradient id={`${clipId}-steel`} x1="0" x2="1">
                    <stop offset="0%" stopColor="#cbd5e1" />
                    <stop offset="38%" stopColor="#f8fafc" />
                    <stop offset="72%" stopColor="#94a3b8" />
                    <stop offset="100%" stopColor="#e2e8f0" />
                </linearGradient>
                <clipPath id={clipId}>
                    <path d="M24 38h96v126c0 14-21 24-48 24s-48-10-48-24z" />
                </clipPath>
            </defs>
            <ellipse cx="72" cy="38" rx="48" ry="18" fill={`url(#${clipId}-steel)`} />
            <path className="process-large-tank-shell" d="M24 38v126c0 14 21 24 48 24s48-10 48-24V38" />
            <rect
                className="process-water-fill"
                x="24"
                y={fillY}
                width="96"
                height={fillHeight}
                clipPath={`url(#${clipId})`}
            />
            <path className="process-water-line" d={`M28 ${fillY}c18 6 70 6 88 0`} clipPath={`url(#${clipId})`} />
            <ellipse className="process-large-tank-rim" cx="72" cy="38" rx="48" ry="18" />
            <rect className="process-tank-value-box" x="42" y="16" width="60" height="24" rx="4" />
            <text className="process-tank-value-text" x="72" y="33">{levelPercentLabel(tag, value)}</text>
            <line className="process-tank-scale" x1="138" x2="138" y1={fillTop} y2={fillBottom} />
            {ticks.map((tick) => {
                const y = fillBottom - (tick / 100) * fillRange;
                return (
                    <g key={tick}>
                        <line className="process-tank-scale" x1="130" x2="138" y1={y} y2={y} />
                        <text className="process-tank-scale-label" x="144" y={y + 4}>{tick}</text>
                    </g>
                );
            })}
        </svg>
    );
}

function StaticTankSymbol() {
    return (
        <svg className="process-static-tank-symbol" viewBox="0 0 180 82" aria-hidden="true">
            <defs>
                <linearGradient id="process-static-tank-steel" x1="0" x2="1">
                    <stop offset="0%" stopColor="#cbd5e1" />
                    <stop offset="45%" stopColor="#f8fafc" />
                    <stop offset="100%" stopColor="#94a3b8" />
                </linearGradient>
            </defs>
            <path d="M22 10h136c13 0 20 16 20 31s-7 31-20 31H22C9 72 2 56 2 41S9 10 22 10z" />
            <line x1="40" x2="40" y1="10" y2="72" />
            <rect x="70" y="30" width="42" height="20" rx="4" />
        </svg>
    );
}

function FlowPumpSymbol({ active, compact = false }) {
    return (
        <svg className={`process-pump-symbol${active ? ' active' : ''}${compact ? ' compact' : ''}`} viewBox="0 0 130 88" aria-hidden="true">
            <rect x="4" y="39" width="28" height="14" rx="3" />
            <circle cx="64" cy="44" r="30" />
            <circle className="process-pump-rotor" cx="64" cy="44" r="15" />
            <path className="process-pump-blades" d="M64 25v38M45 44h38M51 31l26 26M77 31L51 57" />
            <rect x="94" y="36" width="32" height="18" rx="3" />
            <path d="M36 78h56" />
        </svg>
    );
}

function InlineValveSymbol({ active, orientation = 'horizontal' }) {
    return (
        <svg className={`process-inline-valve-symbol ${orientation}${active ? ' active' : ''}`} viewBox="0 0 56 42" aria-hidden="true">
            <path d="M4 8l21 13L4 34zM52 8L31 21l21 13z" />
            <rect x="25" y="15" width="6" height="12" rx="1" />
            <path d="M28 15V4M18 4h20" />
        </svg>
    );
}

function ProcessUnitSymbol({ type }) {
    if (type === 'uv') {
        return (
            <svg className="process-unit-symbol" viewBox="0 0 180 84" aria-hidden="true">
                <rect x="18" y="30" width="144" height="28" rx="9" />
                <path d="M42 44h96M48 8l-12 28M90 4v32M132 8l12 28M48 76L36 52M90 80V52M132 76l12-24" />
            </svg>
        );
    }

    if (type === 'filterUnit') {
        return (
            <svg className="process-unit-symbol" viewBox="0 0 180 86" aria-hidden="true">
                <rect x="14" y="18" width="152" height="50" rx="10" />
                <path d="M34 32h112M34 43h112M34 54h112" />
            </svg>
        );
    }

    if (type === 'roUnit') {
        return (
            <svg className="process-unit-symbol" viewBox="0 0 220 86" aria-hidden="true">
                <rect x="12" y="25" width="196" height="38" rx="19" />
                <path d="M38 44h144M70 29v30M110 29v30M150 29v30" />
            </svg>
        );
    }

    if (type === 'mixer') {
        return (
            <svg className="process-unit-symbol" viewBox="0 0 100 82" aria-hidden="true">
                <rect x="14" y="14" width="72" height="54" rx="8" />
                <path d="M28 28c14 24 30 0 44 24M28 54c14-24 30 0 44-24" />
            </svg>
        );
    }

    if (type === 'outlet') {
        return (
            <svg className="process-unit-symbol" viewBox="0 0 160 64" aria-hidden="true">
                <path d="M8 32h104" />
                <path d="M112 16l36 16-36 16z" />
            </svg>
        );
    }

    return null;
}

function ChemicalTankSymbol() {
    return (
        <svg className="process-chemical-symbol" viewBox="0 0 108 78" aria-hidden="true">
            <rect x="12" y="10" width="84" height="44" rx="5" />
            <path d="M54 54v16" />
            <path d="M44 68h20" />
        </svg>
    );
}

function EquipmentNode({ item, currentData, onOpen }) {
    const value = item.tag ? valueFor(currentData, item.stage, item.tag) : null;
    const active = ['pump', 'dosingPump', 'uv'].includes(item.type) ? isActive(value) : false;
    const clickable = Boolean(item.tag);
    const Component = clickable ? 'button' : 'div';
    const kind = ['pump', 'dosingPump', 'uv'].includes(item.type) ? 'actuator' : 'sensor';

    return (
        <Component
            type={clickable ? 'button' : undefined}
            className={`process-equipment ${item.type}${active ? ' active' : ''}${clickable ? ' clickable' : ''}`}
            style={{ left: item.x, top: item.y, width: item.width, height: item.height }}
            onClick={clickable ? () => onOpen(openableDevice(item.stage, item.tag, kind)) : undefined}
            aria-label={clickable ? `${item.stage} ${item.label} ${item.tag} value ${formatValue(value)}` : undefined}
        >
            <span className="process-equipment-art">
                {item.type === 'largeTank' && <LargeTankSymbol id={item.id} tag={item.tag} value={value} />}
                {item.type === 'staticTank' && <StaticTankSymbol />}
                {(item.type === 'pump' || item.type === 'dosingPump') && (
                    <FlowPumpSymbol active={active} compact={item.type === 'dosingPump'} />
                )}
                {item.type === 'chemicalTank' && <ChemicalTankSymbol />}
                {['mixer', 'filterUnit', 'uv', 'roUnit', 'outlet'].includes(item.type) && (
                    <ProcessUnitSymbol type={item.type} />
                )}
            </span>
            <span className="process-equipment-label">{item.label}</span>
            {item.tag ? (
                <span className="process-equipment-value">
                    {item.tag}: {formatValue(value)}
                </span>
            ) : item.type === 'staticTank' ? (
                <span className="process-equipment-value muted">Static tank</span>
            ) : null}
        </Component>
    );
}

function SensorTag({ sensor, currentData, onOpen }) {
    const value = valueFor(currentData, sensor.stage, sensor.tag);

    return (
        <button
            type="button"
            className="process-sensor-tag"
            style={{ left: sensor.x, top: sensor.y }}
            onClick={() => onOpen(openableDevice(sensor.stage, sensor.tag, 'sensor'))}
            aria-label={`${sensor.stage} sensor ${sensor.label} value ${formatValue(value)}`}
        >
            <span>{sensor.label}</span>
            <strong>{formatValue(value)}</strong>
        </button>
    );
}

function ValveMarker({ valve, currentData, onOpen }) {
    const value = valve.tag ? valueFor(currentData, valve.stage, valve.tag) : null;
    const active = isActive(value);
    const clickable = Boolean(valve.tag);
    const Component = clickable ? 'button' : 'div';

    return (
        <Component
            type={clickable ? 'button' : undefined}
            className={`process-valve-marker ${valve.orientation || 'horizontal'}${active ? ' active' : ''}${clickable ? ' clickable' : ''}`}
            style={{ left: valve.x, top: valve.y }}
            onClick={clickable ? () => onOpen(openableDevice(valve.stage, valve.tag, 'actuator')) : undefined}
            aria-label={clickable ? `${valve.stage} valve ${valve.tag} value ${formatValue(value)}` : undefined}
        >
            <InlineValveSymbol active={active} orientation={valve.orientation} />
        </Component>
    );
}

function ProcessGroup({ group }) {
    return (
        <section
            className={`process-mimic-group${group.labelAlign === 'left' ? ' label-left' : ''}`}
            style={{ left: group.x, top: group.y, width: group.width, height: group.height }}
            aria-label={`${group.label} process area`}
        >
            <span>{group.label}</span>
        </section>
    );
}

function ProcessPipeLayer() {
    return (
        <svg
            className="process-pipe-layer"
            viewBox={`0 0 ${PROCESS_CANVAS.width} ${PROCESS_CANVAS.height}`}
            aria-hidden="true"
        >
            <defs>
                <linearGradient id="process-pipe-steel" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#f8fafc" />
                    <stop offset="45%" stopColor="#94a3b8" />
                    <stop offset="100%" stopColor="#475569" />
                </linearGradient>
            </defs>
            {PROCESS_PIPES.map((pipe) => (
                <path
                    key={pipe.id}
                    className={`process-diagram-pipe ${pipe.kind || 'main'}${pipe.reverse ? ' reverse' : ''}`}
                    d={pipe.d}
                />
            ))}
            {FLOW_ARROWS.map((arrow) => (
                <g
                    key={arrow.id}
                    className="process-flow-arrow"
                    transform={`translate(${arrow.x} ${arrow.y}) rotate(${arrow.rotate})`}
                >
                    <path d="M-6 -5L8 0L-6 5Z" />
                </g>
            ))}
            {FLOW_LABELS.map((label) => (
                <text key={label.text} className="process-flow-label" x={label.x} y={label.y}>{label.text}</text>
            ))}
        </svg>
    );
}

export default function ProcessViewPage() {
    const { stageCurrentData } = useScadaStream();
    const canvasShellRef = useRef(null);
    const [canvasScale, setCanvasScale] = useState(1);
    const [selectedDevice, setSelectedDevice] = useState(null);
    const [deviceHistory, setDeviceHistory] = useState([]);
    const [deviceHistoryLoading, setDeviceHistoryLoading] = useState(false);

    const openDeviceHistory = useCallback((device) => {
        setDeviceHistory([]);
        setDeviceHistoryLoading(true);
        setSelectedDevice(device);
    }, []);

    const closeDeviceHistory = useCallback(() => {
        setSelectedDevice(null);
        setDeviceHistory([]);
        setDeviceHistoryLoading(false);
    }, []);

    useEffect(() => {
        if (!selectedDevice?.stage) {
            return undefined;
        }

        let cancelled = false;
        getHistory({ stage: selectedDevice.stage, limit: DAY_HISTORY_LIMIT }).then((rows) => {
            if (cancelled) return;
            setDeviceHistory(buildDeviceHistory(rows, selectedDevice.field, selectedDevice.kind));
        }).catch(() => {
            if (!cancelled) setDeviceHistory([]);
        }).finally(() => {
            if (!cancelled) setDeviceHistoryLoading(false);
        });

        return () => {
            cancelled = true;
        };
    }, [selectedDevice]);

    useEffect(() => {
        const element = canvasShellRef.current;
        if (!element) {
            return undefined;
        }

        const updateScale = () => {
            const width = element.getBoundingClientRect().width;
            const nextScale = width > 0 ? Math.min(1, width / PROCESS_CANVAS.width) : 1;
            setCanvasScale(nextScale);
        };

        updateScale();
        const resizeObserver = new ResizeObserver(updateScale);
        resizeObserver.observe(element);

        return () => resizeObserver.disconnect();
    }, []);

    return (
        <div className="process-view-page">
            <header className="page-header">
                <div>
                    <h1 className="page-title">Process View</h1>
                    <p className="process-page-subtitle">
                        SWaT process mimic with large dynamic tanks, inline valves, active pumps, and 1-hour device history.
                    </p>
                </div>
            </header>

            <div className="page-container process-page-container">
                <div className="process-toolbar">
                    <div>
                        <strong>SWaT Water Treatment Process Diagram</strong>
                        <span>SCADA mimic focuses on flow, tank level, pumps, and valves. Full device inventory remains available on StagePage.</span>
                    </div>
                    <div className="process-legend" aria-label="Process view legend">
                        <span><i className="legend-dot sensor" /> Sensor tag</span>
                        <span><i className="legend-dot active" /> Active pump/valve</span>
                        <span><i className="legend-dot off" /> Off/static/unknown</span>
                        <span><i className="legend-flow" /> Flow direction</span>
                    </div>
                </div>

                <div
                    className="process-canvas-shell"
                    ref={canvasShellRef}
                    style={{ height: Math.ceil(PROCESS_CANVAS.height * canvasScale) }}
                >
                    <div
                        className="process-canvas process-diagram-canvas"
                        style={{
                            width: PROCESS_CANVAS.width,
                            height: PROCESS_CANVAS.height,
                            transform: `scale(${canvasScale})`,
                        }}
                    >
                        {PROCESS_GROUPS.map((group) => <ProcessGroup key={group.id} group={group} />)}
                        <ProcessPipeLayer />
                        {PROCESS_EQUIPMENT.map((item) => (
                            <EquipmentNode
                                key={item.id}
                                item={item}
                                currentData={stageCurrentData}
                                onOpen={openDeviceHistory}
                            />
                        ))}
                        {PROCESS_SENSORS.map((sensor) => (
                            <SensorTag
                                key={`${sensor.stage}-${sensor.tag}`}
                                sensor={sensor}
                                currentData={stageCurrentData}
                                onOpen={openDeviceHistory}
                            />
                        ))}
                        {PROCESS_VALVES.map((valve) => (
                            <ValveMarker
                                key={valve.id}
                                valve={valve}
                                currentData={stageCurrentData}
                                onOpen={openDeviceHistory}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <DeviceHistoryModal
                device={selectedDevice}
                points={deviceHistory}
                loading={deviceHistoryLoading}
                onClose={closeDeviceHistory}
            />
        </div>
    );
}
