import { acknowledgeAlert } from '../services/api';

function formatTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

export default function AlertCard({ alert, onAcknowledged }) {
    const sev = (alert.severity || 'LOW').toLowerCase();
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
                        {alert.severity || 'WARNING'}
                    </span>
                </div>
                <div className="alert-time">{formatTime(alert.timestamp)}</div>
            </div>

            <div>
                <div className="alert-msg-title">
                    {alert.message ? alert.message.split(' at ')[0] : 'Sự cố bất thường'}
                </div>
                <div className="alert-msg-desc">
                    Value {alert.z_score ? Number(alert.z_score).toFixed(1) : ''} PSI exceeds threshold ({alert.threshold || 2.0} PSI).
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
