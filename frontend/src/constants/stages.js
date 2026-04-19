export const STAGE_CONFIG = {
    P1: {
        threshold: 2.0,
        window: 30,
        name: 'Cấp nước thô',
        monitored: false,
        alertMode: 'monitoring',
        sensors: ['FIT 101', 'LIT 101'],
        actuators: ['P101 Status', 'P102 Status', 'MV 101'],
    },
    P2: {
        threshold: 2.0,
        window: 300,
        name: 'Xử lý hóa chất',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 201', 'AIT 202', 'AIT 203', 'FIT 201', 'LS 201', 'LS 202', 'LSL 203', 'LSLL 203'],
        actuators: ['MV201', 'P201 Status', 'P202 Status', 'P203 Status', 'P204 Status', 'P205 Status', 'P206 Status', 'P207 Status', 'P208 Status'],
    },
    P3: {
        threshold: 2.0,
        window: 30,
        name: 'Siêu lọc',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 301', 'AIT 302', 'AIT 303', 'DPIT 301', 'FIT 301', 'LIT 301'],
        actuators: ['MV 301', 'MV 302', 'MV 303', 'MV 304', 'P301 Status', 'P302 Status'],
    },
    P4: {
        threshold: 2.0,
        window: 30,
        name: 'Khử clo và thẩm thấu ngược',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 401', 'AIT 402', 'FIT 401', 'LIT 401', 'LS 401'],
        actuators: ['UV401', 'P401 Status', 'P402 Status', 'P403 Status', 'P404 Status'],
    },
    P5: {
        threshold: 2.0,
        window: 30,
        name: 'Thu hồi nước sạch',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 501', 'AIT 502', 'AIT 503', 'AIT 504', 'FIT 501', 'FIT 502', 'FIT 503', 'FIT 504', 'PIT 501', 'PIT 502', 'PIT 503'],
        actuators: ['MV 501', 'MV 502', 'MV 503', 'MV 504', 'P501 Status', 'P502 Status'],
    },
    P6: {
        threshold: 2.0,
        window: 30,
        name: 'Làm sạch hệ thống',
        monitored: false,
        alertMode: 'monitoring',
        sensors: ['FIT 601', 'LSH 601', 'LSH 602', 'LSH 603', 'LSL 601', 'LSL 602', 'LSL 603'],
        actuators: ['P601 Status', 'P602 Status', 'P603 Status'],
    },
};

export const STAGES = Object.keys(STAGE_CONFIG);
export const MONITORED_STAGES = STAGES.filter((stage) => STAGE_CONFIG[stage].monitored);
export const MONITORING_ONLY_STAGES = STAGES.filter(
    (stage) => STAGE_CONFIG[stage].alertMode === 'monitoring'
);
