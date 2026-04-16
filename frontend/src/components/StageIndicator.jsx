export default function StageIndicator({ stage, name, isCritical, isWarming = false, isMonitored = true }) {
    const statusClass = isCritical ? 'critical' : isWarming ? 'warning' : 'normal';
    const iconName = !isMonitored ? 'monitoring' : isCritical ? 'warning' : isWarming ? 'schedule' : 'check_circle';

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
                        MONITORING ONLY
                    </span>
                )}
                {isMonitored && isWarming && !isCritical && (
                    <span className="stage-critical-text" style={{ color: 'var(--color-medium)' }}>
                        WARMING UP
                    </span>
                )}
                {isMonitored && isCritical && <span className="stage-critical-text">DANGER</span>}
            </div>
        </div>
    );
}
