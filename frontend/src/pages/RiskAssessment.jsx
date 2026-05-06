import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '../stores/useStore'
import StatCard from '../components/StatCard'
import RiskHeatmap from '../components/RiskHeatmap'
import { AlertTriangle, Flame, Droplet, Truck } from 'lucide-react'

function RiskAssessment() {
  const { fetchRiskAssessment, riskData } = useStore()
  const [selectedRegion] = useState('REGION_001')

  useEffect(() => {
    fetchRiskAssessment(selectedRegion)
  }, [fetchRiskAssessment, selectedRegion])

  const scores = riskData?.risk_scores || {}
  const riskValue = (key, fallback) => `${((scores[key] ?? fallback) * 10).toFixed(1)}/10`

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-white mb-2">Risk Assessment</h1>
        <p className="text-slate-400">Comprehensive threat analysis and vulnerability mapping</p>
      </motion.div>

      {/* Risk Scores Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
      >
        <motion.div variants={itemVariants}>
          <StatCard
            icon={Flame}
            label="Fire Risk"
            value={riskValue('fire_risk', 0.75)}
            trend={5}
            color="red"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={Droplet}
            label="Flood Risk"
            value={riskValue('flood_risk', 0.45)}
            trend={2}
            color="blue"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={Truck}
            label="Accessibility"
            value={riskValue('accessibility_risk', 0.55)}
            trend={-1}
            color="yellow"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={AlertTriangle}
            label="Overall Risk"
            value={riskValue('overall_risk', 0.62)}
            trend={3}
            color="red"
          />
        </motion.div>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Heatmap */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <div className="glass-dark p-6 rounded-lg">
            <h2 className="text-xl font-semibold text-white mb-4">Risk Heatmap</h2>
            <RiskHeatmap riskData={riskData} />
          </div>
        </motion.div>

        {/* Risk Factors */}
        <motion.div variants={itemVariants}>
          <div className="glass-dark p-6 rounded-lg h-full">
            <h3 className="text-white font-semibold mb-4">Contributing Factors</h3>
            <div className="space-y-3 text-sm">
              {[
                { factor: 'High Population Density', severity: 95 },
                { factor: 'Poor Road Access', severity: 78 },
                { factor: 'Low Elevation', severity: 62 },
                { factor: 'Flammable Roofing', severity: 85 },
                { factor: 'Limited Water Supply', severity: 72 }
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-300">{item.factor}</span>
                    <span className="text-cyan-400 font-bold text-xs">{item.severity}%</span>
                  </div>
                  <div className="w-full bg-slate-700/30 rounded-full h-1.5">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.severity}%` }}
                      transition={{ duration: 1, delay: 0.3 }}
                      className={`h-full rounded-full ${
                        item.severity > 80 ? 'bg-red-500' :
                        item.severity > 60 ? 'bg-orange-500' :
                        'bg-yellow-500'
                      }`}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Risk Timeline */}
      <motion.div
        variants={itemVariants}
        initial="hidden"
        animate="visible"
        transition={{ delay: 0.4 }}
        className="mt-8 glass-dark p-6 rounded-lg"
      >
        <h2 className="text-xl font-semibold text-white mb-6">Risk Trend (12 Months)</h2>
        <div className="space-y-4">
          {Array.from({ length: 12 }).map((_, i) => {
            const month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i]
            const riskTrend = 0.5 + (i * 0.02)
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-center gap-4"
              >
                <span className="text-slate-400 w-12 text-sm">{month}</span>
                <div className="flex-1 h-6 bg-slate-700/30 rounded-lg overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${riskTrend * 100}%` }}
                    transition={{ duration: 1, delay: 0.3 }}
                    className="h-full bg-gradient-to-r from-yellow-500 to-red-500"
                  />
                </div>
                <span className="text-cyan-400 text-sm font-bold w-12 text-right">{(riskTrend * 100).toFixed(0)}%</span>
              </motion.div>
            )
          })}
        </div>
      </motion.div>
    </div>
  )
}

export default RiskAssessment
