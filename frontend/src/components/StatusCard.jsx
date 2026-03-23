export default function StatusCard({ icon, iconColor, label, value, sub, valColor }) {
    return (
        <div className="card status-card">
            <div style={{ flex: 1 }}>
                <div className="status-card-label">{label}</div>
                <div className={`status-card-val-badge ${valColor}`}>
                    {value}
                </div>
                <div className="status-card-sub">{sub}</div>
            </div>
            <div className={`status-card-icon ${iconColor}`}>
                <span className="material-symbols-outlined">{icon}</span>
            </div>
        </div>
    );
}
