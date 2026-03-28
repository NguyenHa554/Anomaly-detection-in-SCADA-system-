export default function StageIndicator({ stage, name, isCritical, isMonitored = true }) {
    const statusClass = isCritical ? 'critical' : 'normal';
    const iconName = !isMonitored ? 'block' : isCritical ? 'warning' : 'check_circle';

    return (
        <div className={`stage-node ${statusClass}`}>
            <div className="stage-node-icon-wrap">
                <span className="material-symbols-outlined" style={{ fontSize: isCritical ? 24 : 20 }}>
                    {iconName}
                </span>
            </div>
            <div className="stage-info-box">
                <div className="stage-p-label">{stage}</div>
                <div className="stage-desc">{name}</div>
                {!isMonitored && (
                    <span className="stage-critical-text" style={{ color: 'var(--text-muted)' }}>
                        KHÔNG GIÁM SÁT AI
                    </span>
                )}
                {isMonitored && isCritical && <span className="stage-critical-text">BẤT THƯỜNG</span>}
            </div>
        </div>
    );
}
