import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ProcessViewPage from './pages/ProcessViewPage';
import StagePage from './pages/StagePage';
import AlertsPage from './pages/AlertsPage';
import UploadPage from './pages/UploadPage';
import SettingsPage from './pages/SettingsPage';
import ReportsPage from './pages/ReportsPage';
import { ScadaStreamProvider } from './context/ScadaStreamContext';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <ScadaStreamProvider>
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/process" element={<ProcessViewPage />} />
              <Route path="/stages/:stageId" element={<StagePage />} />
              <Route path="/incidents" element={<AlertsPage />} />
              <Route path="/reports" element={<ReportsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </ScadaStreamProvider>
    </BrowserRouter>
  );
}
