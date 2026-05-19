import SensorChart from './SensorChart';
import {
    DEVICE_HISTORY_GAP_THRESHOLD_MS,
    DEVICE_HISTORY_WINDOW_MS,
} from '../services/deviceHistory';
import { formatBackendDateTime } from '../utils/time';

function formatDetailTimestamp(timestamp) {
    return formatBackendDateTime(timestamp);
}

export default function DeviceHistoryModal({ device, points, loading, onClose }) {
    if (!device) {
        return null;
    }

    const latestPoint = points.at(-1);
    const minValue = points.length ? Math.min(...points.map((point) => point.value)) : null;
    const maxValue = points.length ? Math.max(...points.map((point) => point.value)) : null;
    const deviceTypeLabel = device.kind === 'actuator' ? 'Actuator' : 'Sensor';

    return (
        <div className="device-modal-backdrop" role="presentation" onClick={onClose}>
            <section
                className="device-modal"
                role="dialog"
                aria-modal="true"
                aria-label={`${device.field} 1 hour history`}
                onClick={(event) => event.stopPropagation()}
            >
                <div className="device-modal-header">
                    <div>
                        <div className="device-modal-eyebrow">{deviceTypeLabel} history</div>
                        <h2 className="device-modal-title">{device.field}</h2>
                        <div className="device-modal-subtitle">
                            Last 1 hour from stored stage history{device.stage ? ` | ${device.stage}` : ''}
                        </div>
                    </div>
                    <button className="device-modal-close" type="button" onClick={onClose} aria-label="Close device history">
                        <span className="material-symbols-outlined">close</span>
                    </button>
                </div>

                <div className="device-modal-stats">
                    <div>
                        <span>Latest</span>
                        <strong>{latestPoint ? latestPoint.value.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                        <span>Min</span>
                        <strong>{minValue != null ? minValue.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                        <span>Max</span>
                        <strong>{maxValue != null ? maxValue.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                        <span>Samples</span>
                        <strong>{points.length}</strong>
                    </div>
                </div>

                <div className="device-modal-chart">
                    {loading ? (
                        <div className="device-modal-empty">Loading device history...</div>
                    ) : points.length > 0 ? (
                        <SensorChart
                            data={points}
                            threshold={null}
                            height={420}
                            dataKey="value"
                            windowMs={DEVICE_HISTORY_WINDOW_MS}
                            tickStepMs={5 * 60 * 1000}
                            minTickPx={68}
                            showMiniOverview
                            resetKey={`modal-${device.stage || 'stage'}-${device.field}`}
                            gapThresholdMs={DEVICE_HISTORY_GAP_THRESHOLD_MS}
                            aggregation="avg"
                            aggregationIntervalMs={1000}
                        />
                    ) : (
                        <div className="device-modal-empty">No stored values for this device</div>
                    )}
                </div>

                <div className="device-modal-table-wrap">
                    <table className="device-modal-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Value</th>
                                <th>Window state</th>
                            </tr>
                        </thead>
                        <tbody>
                            {points.slice(-300).reverse().map((point) => (
                                <tr key={`${point.ts}-${point.value}`}>
                                    <td>{formatDetailTimestamp(point.ts)}</td>
                                    <td>{point.value.toFixed(4)}</td>
                                    <td>{point.isAnomaly ? 'DANGER' : 'NORMAL'}</td>
                                </tr>
                            ))}
                            {!loading && points.length === 0 && (
                                <tr>
                                    <td colSpan="3">No rows available</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
}
