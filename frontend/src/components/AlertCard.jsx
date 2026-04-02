import { acknowledgeAlert } from '../services/api';

function formatTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

export default function AlertCard({ alert, onAcknowledged }) {
    const sev = (alert.severity || 'DANGER').toLowerCase();
    const acked = alert.acknowledged;

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
                    {alert.message ? alert.message.split(' at ')[0] : 'Sự cố bất thường'}
                </div>
                <div className="alert-msg-desc">
                    Confirmed episode start. Max z-score {alert.max_z_score != null ? Number(alert.max_z_score).toFixed(2) : '—'} over threshold ({alert.threshold || 2.0}).
                </div>
            </div>

            {!acked && (
                <button className="btn-acknowledge" onClick={handleAck}>
                    Xác nhận xử lý
                </button>
            )}
        </div>
    );
}
