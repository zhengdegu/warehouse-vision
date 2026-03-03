import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Sidebar } from './components/Sidebar'
import { Statusbar } from './components/Statusbar'
import { LivePage } from './pages/LivePage'
import { EventsPage } from './pages/EventsPage'
import { ConfigPage } from './pages/ConfigPage'
import { TrainingPage } from './pages/TrainingPage'
import { SystemPage } from './pages/SystemPage'
import { AccuracyPage } from './pages/AccuracyPage'
import { WebSocketProvider } from './hooks/useWebSocket'

export default function App() {
  return (
    <BrowserRouter>
      <WebSocketProvider>
        <Sidebar />
        <div className="flex flex-col flex-1 min-w-0 h-screen">
          <main className="flex-1 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Navigate to="/live" replace />} />
              <Route path="/live" element={<LivePage />} />
              <Route path="/events" element={<EventsPage />} />
              <Route path="/accuracy" element={<AccuracyPage />} />
              <Route path="/config" element={<ConfigPage />} />
              <Route path="/training" element={<TrainingPage />} />
              <Route path="/system" element={<SystemPage />} />
            </Routes>
          </main>
          <Statusbar />
        </div>
      </WebSocketProvider>
    </BrowserRouter>
  )
}
