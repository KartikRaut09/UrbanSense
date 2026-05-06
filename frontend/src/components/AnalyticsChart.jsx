import React from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const chartData = [
  { month: 'Jan', slumArea: 2.5, population: 45000, fireRisk: 0.65 },
  { month: 'Feb', slumArea: 2.65, population: 48000, fireRisk: 0.68 },
  { month: 'Mar', slumArea: 2.8, population: 51000, fireRisk: 0.72 },
  { month: 'Apr', slumArea: 3.0, population: 55000, fireRisk: 0.75 },
  { month: 'May', slumArea: 3.2, population: 58000, fireRisk: 0.78 },
  { month: 'Jun', slumArea: 3.5, population: 62000, fireRisk: 0.82 },
]

function AnalyticsChart() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full bg-slate-800/30 backdrop-blur-xl border border-slate-700/30 rounded-lg p-6"
    >
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-white mb-2">Growth Trends</h3>
        <p className="text-slate-400 text-sm">Slum area expansion and risk trend over time</p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <defs>
            <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.5)" />
          <XAxis dataKey="month" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              borderRadius: '8px',
              color: '#e2e8f0'
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="slumArea"
            stroke="#06b6d4"
            strokeWidth={2}
            dot={{ fill: '#06b6d4', r: 4 }}
            activeDot={{ r: 6 }}
            name="Area (km²)"
          />
          <Line
            type="monotone"
            dataKey="fireRisk"
            stroke="#ef4444"
            strokeWidth={2}
            dot={{ fill: '#ef4444', r: 4 }}
            name="Fire Risk"
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  )
}

export default AnalyticsChart
