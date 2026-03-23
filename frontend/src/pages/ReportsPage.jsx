import { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
    PieChart, Pie, Cell, AreaChart, Area, ReferenceLine
} from 'recharts';

// Mock Data for Charts
const uptimeData = Array.from({ length: 30 }, (_, i) => ({
    day: `Mar ${i + 1}`,
    uptime: 95 + Math.random() * 5
}));

const incidentData = [
    { name: 'Giai đoạn P1', value: 45 },
    { name: 'Giai đoạn P4', value: 30 },
    { name: 'Giai đoạn P3', value: 15 },
    { name: 'Khác', value: 10 },
];
const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#94a3b8'];

const zScoreData = Array.from({ length: 50 }, (_, i) => {
    const z = (i / 10) - 1; // -1 to 4
    // Simple bell curve approx
    const count = 1000 * Math.exp(-(Math.pow(z - 0.5, 2)) / 0.5);
    return { z: z.toFixed(1), count: Math.round(count) };
});

export default function ReportsPage() {
    const [activeTab, setActiveTab] = useState('health');

    const handlePrint = () => {
        window.print();
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '0 32px' }}>
            {/* Top Control Bar */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '24px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                <div>
                    <h1 className="page-title" style={{ fontSize: '1.5rem', marginBottom: 8 }}>Báo cáo Phân tích</h1>
                    <div style={{ display: 'flex', gap: 16 }}>
                        <select className="form-select" style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}>
                            <option>Tháng này</option>
                            <option>Hôm qua</option>
                            <option>Tuần trước</option>
                            <option>Tùy chỉnh...</option>
                        </select>
                        <select className="form-select" style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}>
                            <option>Toàn bộ nhà máy</option>
                            <option>Stage P1 - Cấp nước thô</option>
                            <option>Stage P2 - Xử lý hóa chất</option>
                            <option>Stage P3 - Siêu lọc</option>
                            <option>Stage P4 - Khử clo</option>
                            <option>Stage P5 - Thu hồi</option>
                            <option>Stage P6 - Làm sạch</option>
                        </select>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: 12 }}>
                    <button 
                        onClick={handlePrint}
                        style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 20px', background: 'var(--accent-primary)', color: '#fff', border: 'none', borderRadius: 8, fontWeight: 600, cursor: 'pointer' }}
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: 20 }}>picture_as_pdf</span>
                        Xuất PDF
                    </button>
                    <button style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 20px', background: 'var(--bg-surface)', color: 'var(--accent-primary)', border: '1px solid var(--accent-primary)', borderRadius: 8, fontWeight: 600, cursor: 'pointer' }}>
                        <span className="material-symbols-outlined" style={{ fontSize: 20 }}>download</span>
                        Xuất Excel
                    </button>
                </div>
            </div>

            <div className="page-container" style={{ padding: '24px 0', maxWidth: 1400, margin: '0 auto', width: '100%' }}>
                
                {/* Auto Summary Section */}
                <div className="card" style={{ padding: 24, marginBottom: 24 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                        <span className="material-symbols-outlined" style={{ color: 'var(--color-medium)', fontSize: 28 }}>lightbulb</span>
                        <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Tổng quan Vận hành AI</h2>
                    </div>
                    <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                        Trong tháng 3/2026, hệ thống duy trì <strong>99.2%</strong> uptime. Động cơ AI giám sát đã phát hiện <strong>128 sự cố</strong>, 
                        tập trung chủ yếu tại Giai đoạn P1 (Cấp nước thô) và P4. Thời gian phản hồi trung bình (MTTR) đạt 1m 45s, 
                        <span style={{ color: 'var(--color-normal)', fontWeight: 600 }}> cải thiện 12%</span> so với tháng trước.
                    </p>
                </div>

                {/* Tabs */}
                <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', marginBottom: 24 }}>
                    <button 
                        onClick={() => setActiveTab('health')}
                        style={{ padding: '12px 24px', background: 'none', border: 'none', borderBottom: activeTab === 'health' ? '3px solid var(--accent-primary)' : '3px solid transparent', color: activeTab === 'health' ? 'var(--accent-primary)' : 'var(--text-muted)', fontWeight: 600, fontSize: '1rem', cursor: 'pointer' }}
                    >
                        Sức khỏe Vận hành
                    </button>
                    <button 
                        onClick={() => setActiveTab('security')}
                        style={{ padding: '12px 24px', background: 'none', border: 'none', borderBottom: activeTab === 'security' ? '3px solid var(--accent-primary)' : '3px solid transparent', color: activeTab === 'security' ? 'var(--accent-primary)' : 'var(--text-muted)', fontWeight: 600, fontSize: '1rem', cursor: 'pointer' }}
                    >
                        An ninh & Sự cố
                    </button>
                    <button 
                        onClick={() => setActiveTab('ai')}
                        style={{ padding: '12px 24px', background: 'none', border: 'none', borderBottom: activeTab === 'ai' ? '3px solid var(--accent-primary)' : '3px solid transparent', color: activeTab === 'ai' ? 'var(--accent-primary)' : 'var(--text-muted)', fontWeight: 600, fontSize: '1rem', cursor: 'pointer' }}
                    >
                        Hiệu suất AI
                    </button>
                </div>

                {/* Tab Content */}
                {activeTab === 'health' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
                        <div className="card" style={{ padding: 24 }}>
                            <h3 className="card-title" style={{ marginBottom: 20 }}>Uptime Hệ thống</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={uptimeData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)"/>
                                    <XAxis dataKey="day" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                                    <YAxis domain={[90, 100]} tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                                    <RechartsTooltip cursor={{fill: 'var(--bg-base)'}} />
                                    <Bar dataKey="uptime" fill="var(--color-normal)" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <div className="card" style={{ padding: 24 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Thống kê Thiết bị Chấp hành</h3>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                    {[
                                        {name: 'P101 Pump', pct: 85},
                                        {name: 'P203 Pump', pct: 62},
                                        {name: 'MV201 Valve', pct: 45}
                                    ].map(eq => (
                                        <div key={eq.name}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 4 }}>
                                                <span>{eq.name}</span>
                                                <span style={{ fontWeight: 600 }}>{eq.pct}%</span>
                                            </div>
                                            <div style={{ width: '100%', height: 8, background: 'var(--bg-base)', borderRadius: 4 }}>
                                                <div style={{ width: `${eq.pct}%`, height: '100%', background: 'var(--accent-primary)', borderRadius: 4 }}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div className="card" style={{ padding: 24, flex: 1 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Top Cảm biến Biến thiên</h3>
                                <table style={{ width: '100%', fontSize: '0.85rem', textAlign: 'left', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                                            <th style={{ paddingBottom: 8 }}>Cảm biến</th>
                                            <th style={{ paddingBottom: 8 }}>Độ lệch</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td style={{ paddingTop: 8, fontWeight: 600}}>LIT 101</td><td style={{ paddingTop: 8}}>Hơi cao</td></tr>
                                        <tr><td style={{ paddingTop: 8, fontWeight: 600}}>FIT 401</td><td style={{ paddingTop: 8}}>Rất cao</td></tr>
                                        <tr><td style={{ paddingTop: 8, fontWeight: 600}}>AIT 502</td><td style={{ paddingTop: 8}}>Bình thường</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'security' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24 }}>
                        <div className="card" style={{ padding: 24, gridColumn: 'span 2' }}>
                            <h3 className="card-title" style={{ marginBottom: 20 }}>Phân bố Cảnh báo theo Giai đoạn</h3>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={incidentData}
                                            cx="50%" cy="50%"
                                            innerRadius={80} outerRadius={120}
                                            paddingAngle={2}
                                            dataKey="value"
                                        >
                                            {incidentData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <RechartsTooltip />
                                    </PieChart>
                                </ResponsiveContainer>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginRight: 40 }}>
                                    {incidentData.map((d, i) => (
                                        <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                            <div style={{ width: 12, height: 12, borderRadius: '50%', background: COLORS[i] }} />
                                            <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{d.name} ({d.value}%)</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <div className="card" style={{ padding: 24, textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', flex: 1 }}>
                                <h3 className="card-title" style={{ justifyContent: 'center', marginBottom: 16 }}>Thời gian Phản hồi (MTTR)</h3>
                                <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: 8 }}>1m 45s</div>
                                <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 4, color: 'var(--color-normal)', fontSize: '0.9rem', fontWeight: 600 }}>
                                    <span className="material-symbols-outlined" style={{ fontSize: 18 }}>trending_down</span>
                                    Cải thiện 12%
                                </div>
                            </div>
                            <div className="card" style={{ padding: 24, flex: 1 }}>
                                <h3 className="card-title" style={{ marginBottom: 16 }}>Bản đồ Nhiệt (Heatmap)</h3>
                                <div style={{ width: '100%', height: 120, background: 'linear-gradient(90deg, #fef2f2 0%, #ef4444 50%, #7f1d1d 100%)', borderRadius: 8, display: 'flex', alignItems: 'flex-end', padding: 8 }}>
                                    <span style={{ color: '#fff', fontSize: '0.75rem', fontWeight: 600 }}>Tần suất cao nhất lúc 15:00</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'ai' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 24 }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                            <div className="card" style={{ padding: 24, textAlign: 'center' }}>
                                <h3 className="card-title" style={{ justifyContent: 'center', marginBottom: 16 }}>Tổng Dữ liệu Đã Xử lý</h3>
                                <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-primary)', marginBottom: 8 }}>2,592,000</div>
                                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Mẫu cảm biến được phân tích</div>
                            </div>
                            <div className="card" style={{ padding: 24, textAlign: 'center' }}>
                                <h3 className="card-title" style={{ justifyContent: 'center', marginBottom: 16 }}>Tỷ lệ Cảnh báo Giả (False Alarm)</h3>
                                <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--color-normal)' }}>2.1%</div>
                            </div>
                        </div>

                        <div className="card" style={{ padding: 24 }}>
                            <h3 className="card-title" style={{ marginBottom: 20 }}>Phân phối Z-Score & Ngưỡng An Toàn</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={zScoreData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)"/>
                                    <XAxis dataKey="z" tick={{fontSize: 12}} />
                                    <YAxis hide />
                                    <RechartsTooltip />
                                    <ReferenceLine x="2.0" stroke="var(--color-critical)" strokeDasharray="4 4" label={{ value: 'T=2.0 Threshold', position: 'top', fill: 'var(--color-critical)', fontSize: 12 }} />
                                    <Area type="monotone" dataKey="count" stroke="var(--accent-primary)" fill="var(--accent-glow)" />
                                </AreaChart>
                            </ResponsiveContainer>
                            <p style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 16 }}>Hầu hết dữ liệu vận hành phân bổ trong vùng an toàn (Z &lt; 2.0). Các chuỗi vượt ngưỡng liên tục sẽ kích hoạt cảnh báo.</p>
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
}
