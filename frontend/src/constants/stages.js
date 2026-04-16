export const STAGE_CONFIG = {
    P1: {
        threshold: 2.0,
        window: 30,
        name: 'Cap nuoc tho',
        monitored: false,
        alertMode: 'monitoring',
        sensors: ['FIT 101', 'LIT 101'],
        actuators: ['P101 Status', 'MV 101'],
    },
    P2: {
        threshold: 2.0,
        window: 300,
        name: 'Xu ly hoa chat',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 201', 'AIT 202', 'FIT 201'],
        actuators: ['P203 Status', 'MV201'],
    },
    P3: {
        threshold: 2.0,
        window: 30,
        name: 'Sieu loc',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 301', 'DPIT 301', 'FIT 301', 'LIT 301'],
        actuators: ['P301 Status', 'MV 301'],
    },
    P4: {
        threshold: 2.0,
        window: 30,
        name: 'Khu clo va tham thau nguoc',
        monitored: true,
        alertMode: 'production',
        sensors: ['AIT 401', 'AIT 402', 'FIT 401', 'LIT 401', 'LS 401'],
        actuators: ['P401 Status', 'P402 Status', 'P403 Status', 'P404 Status', 'UV401'],
    },
    P5: {
        threshold: 2.0,
        window: 30,
        name: 'Thu hoi nuoc sach',
        monitored: true,
        alertMode: 'production',
        sensors: ['FIT 501', 'PIT 501', 'AIT 501'],
        actuators: ['P501 Status', 'MV 501'],
    },
    P6: {
        threshold: 2.0,
        window: 30,
        name: 'Lam sach he thong',
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
