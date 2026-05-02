import React from 'react'
import { motion } from 'framer-motion'
import StatCard from '../components/StatCard'
import AnalyticsChart from '../components/AnalyticsChart'
import { BarChart3 } from 'lucide-react'

function Analysis() {
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
        <h1 className="text-4xl font-bold text-white mb-2">Detailed Analysis</h1>
        <p className="text-slate-400">In-depth urban metrics and infrastructure analysis</p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
      >
        <motion.div variants={itemVariants}>
          <StatCard
            icon={BarChart3}
            label="Building Density"
            value="850"
            trend={12}
            color="cyan"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={BarChart3}
            label="Infrastructure"
            value="58%"
            trend={-3}
            color="green"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={BarChart3}
            label="Livability Index"
            value="4.2/10"
            trend={2}
            color="yellow"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={BarChart3}
            label="Development Score"
            value="6.8/10"
            trend={5}
            color="blue"
          />
        </motion.div>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <motion.div variants={itemVariants}>
          <AnalyticsChart />
        </motion.div>

        <motion.div variants={itemVariants}>
          <div className="glass-dark p-6 rounded-lg">
            <h2 className="text-xl font-semibold text-white mb-4">Area Breakdown</h2>
            <div className="space-y-4">
              {[
                { label: 'Residential', value: 65, color: 'from-blue-500 to-cyan-500' },
                { label: 'Commercial', value: 15, color: 'from-purple-500 to-blue-500' },
                { label: 'Industrial', value: 12, color: 'from-orange-500 to-yellow-500' },
                { label: 'Open Space', value: 8, color: 'from-green-500 to-emerald-500' }
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: idx * 0.1 }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-300">{item.label}</span>
                    <span className="text-cyan-400 font-bold">{item.value}%</span>
                  </div>
                  <div className="w-full bg-slate-700/30 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.value}%` }}
                      transition={{ duration: 1, delay: 0.3 }}
                      className={`h-full rounded-full bg-gradient-to-r ${item.color}`}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default Analysis
