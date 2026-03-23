export default function StageIndicator({ stage, name, isCritical }) {
    const statusClass = isCritical ? 'critical' : 'normal';

    return (
        <div className={`stage-node ${statusClass}`}>
            <div className="stage-node-icon-wrap">
                {isCritical ? (
                    <span className="material-symbols-outlined" style={{ fontSize: 24 }}>warning</span>
                ) : (
                    <span className="material-symbols-outlined" style={{ fontSize: 20 }}>check_circle</span>
                )}
            </div>
            <div className="stage-info-box">
                <div className="stage-p-label">{stage}</div>
                <div className="stage-desc">{name}</div>
                {isCritical && <span className="stage-critical-text">BẤT THƯỜNG</span>}
            </div>
        </div>
    );
}
