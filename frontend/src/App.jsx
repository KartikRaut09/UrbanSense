import React, { Suspense } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Analysis from './pages/Analysis'
import RiskAssessment from './pages/RiskAssessment'
import Insights from './pages/Insights'
import Settings from './pages/Settings'

function App() {
  return (
    <Router>
      <div className="w-full h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
        <Navbar />
        <main className="pt-16 h-full overflow-y-auto">
          <Suspense fallback={<LoadingScreen />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/risk" element={<RiskAssessment />} />
              <Route path="/insights" element={<Insights />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </Router>
  )
}

function LoadingScreen() {
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-slate-300">Loading UrbanSense...</p>
      </div>
    </div>
  )
}

export default App
