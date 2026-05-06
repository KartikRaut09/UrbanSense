import React, { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Flame, Droplet, Truck } from 'lucide-react'

function RiskHeatmapComponent({ riskData }) {
  const [activeRisk, setActiveRisk] = useState('overall')
  const scores = riskData?.risk_scores || {}

  const riskTypes = [
    { id: 'fire', label: 'Fire Risk', icon: Flame, color: 'from-red-500 to-orange-500', value: scores.fire_risk ?? 0.65 },
    { id: 'flood', label: 'Flood Risk', icon: Droplet, color: 'from-blue-500 to-cyan-500', value: scores.flood_risk ?? 0.45 },
    { id: 'access', label: 'Accessibility', icon: Truck, color: 'from-yellow-500 to-orange-500', value: scores.accessibility_risk ?? 0.55 },
  ]
  const activeValue = activeRisk === 'overall'
    ? scores.overall_risk ?? 0.6
    : riskTypes.find((risk) => risk.id === activeRisk)?.value ?? 0.6
  const heatmapCells = useMemo(() => (
    Array.from({ length: 64 }, (_, i) => {
      const wave = Math.sin((i + 1) * 1.7) * 0.16
      const rowBias = (Math.floor(i / 8) - 3.5) * 0.018
      return Math.min(1, Math.max(0.05, activeValue + wave + rowBias))
    })
  ), [activeValue])

  return (
    <div className="space-y-6">
      {/* Risk Tabs */}
      <div className="flex gap-2 flex-wrap">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setActiveRisk('overall')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            activeRisk === 'overall'
              ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-500/50'
              : 'bg-slate-700/30 text-slate-300 border border-slate-600/30'
          }`}
        >
          Overall Risk
        </motion.button>
        {riskTypes.map((risk) => (
          <motion.button
            key={risk.id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setActiveRisk(risk.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
              activeRisk === risk.id
                ? `bg-gradient-to-r ${risk.color} text-white border border-white/30`
                : 'bg-slate-700/30 text-slate-300 border border-slate-600/30'
            }`}
          >
            <risk.icon size={16} />
            {risk.label}
          </motion.button>
        ))}
      </div>

      {/* Heatmap Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-8 gap-1 bg-slate-800/30 p-4 rounded-lg border border-slate-700/30"
      >
        <AnimatePresence>
          {heatmapCells.map((riskValue, i) => {
            const getColor = () => {
              if (riskValue > 0.7) return 'bg-red-600 hover:bg-red-500'
              if (riskValue > 0.5) return 'bg-orange-500 hover:bg-orange-400'
              if (riskValue > 0.3) return 'bg-yellow-500 hover:bg-yellow-400'
              return 'bg-green-600 hover:bg-green-500'
            }

            return (
              <motion.div
                key={i}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: i * 0.02 }}
                whileHover={{ scale: 1.2, zIndex: 10 }}
                className={`aspect-square rounded cursor-pointer transition-all ${getColor()}`}
                title={`Risk Level: ${(riskValue * 100).toFixed(0)}%`}
              />
            )
          })}
        </AnimatePresence>
      </motion.div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-8 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-600 rounded" />
          <span className="text-slate-400">Low</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-500 rounded" />
          <span className="text-slate-400">Moderate</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-orange-500 rounded" />
          <span className="text-slate-400">High</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-600 rounded" />
          <span className="text-slate-400">Critical</span>
        </div>
      </div>
    </div>
  )
}

export default RiskHeatmapComponent
