import { acknowledgeAlert } from '../services/api';
import { formatBackendTime } from '../utils/time';

function formatTime(ts) {
    if (!ts) return '--';
    return formatBackendTime(ts, { hour: '2-digit', minute: '2-digit' });
}

export default function AlertCard({ alert, onAcknowledged }) {
    const sev = (alert.severity || 'DANGER').toLowerCase();
    const acked = alert.acknowledged;
    const affectedStages = alert.affected_stages || (alert.stage ? [alert.stage] : []);

    async function handleAck() {
        try {
            await acknowledgeAlert(alert.id);
            onAcknowledged?.(alert.id);
        } catch (err) {
            console.error('Acknowledge failed:', err);
        }
    }

    return (
        <div className="alert-card" style={{ opacity: acked ? 0.6 : 1 }}>
            <div className="alert-card-header">
                <div>
                    <span className={`severity-badge ${sev}`}>
                        {alert.severity || 'DANGER'}
                    </span>
                </div>
                <div className="alert-time">{formatTime(alert.created_at || alert.timestamp)}</div>
            </div>

            <div>
                <div className="alert-msg-title">
                    {alert.incident_id ? `Incident #${alert.incident_id}` : 'Anomaly incident'}
                </div>
                <div className="alert-msg-desc">
                    Primary {alert.primary_stage || alert.stage || '--'}. Affected: {affectedStages.join(', ') || '--'}.
                    {' '}Max z-score {alert.max_z_score != null ? Number(alert.max_z_score).toFixed(2) : '--'}.
                </div>
            </div>

            {!acked && (
                <button className="btn-acknowledge" onClick={handleAck}>
                    Acknowledge
                </button>
            )}
        </div>
    );
}
