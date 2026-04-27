import AlertCard from './AlertCard';

export default function AlertPanel({ alerts = [], onAcknowledged }) {
    return (
        <>
            <div className="alert-list" style={{ padding: '0 16px 16px' }}>
                {alerts.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
                        <span className="material-symbols-outlined" style={{ fontSize: 32, opacity: 0.5 }}>check_circle</span>
                        <div style={{ fontSize: '0.85rem', marginTop: 8 }}>Không có cảnh báo mới</div>
                    </div>
                ) : (
                    alerts.map(a => (
                        <AlertCard key={a.id} alert={a} onAcknowledged={onAcknowledged} />
                    ))
                )}
            </div>
        </>
    );
}
