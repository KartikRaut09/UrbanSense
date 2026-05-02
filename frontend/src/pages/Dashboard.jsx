import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '../stores/useStore'
import StatCard from '../components/StatCard'
import AnalyticsChart from '../components/AnalyticsChart'
import InteractiveMap from '../components/InteractiveMap'
import { Activity, MapPin, AlertTriangle, Users } from 'lucide-react'

function Dashboard() {
  const { fetchRegions, regions, loading } = useStore()

  useEffect(() => {
    fetchRegions()
  }, [])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-white mb-2">
          Urban Intelligence Dashboard
        </h1>
        <p className="text-slate-400">Real-time slum mapping and risk assessment</p>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
      >
        <motion.div variants={itemVariants}>
          <StatCard
            icon={MapPin}
            label="Active Regions"
            value={regions.length || 0}
            trend={12}
            color="cyan"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={Users}
            label="Total Population"
            value="3.2M"
            trend={8}
            color="green"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={AlertTriangle}
            label="Critical Zones"
            value={12}
            trend={3}
            color="red"
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard
            icon={Activity}
            label="Avg Risk Score"
            value="6.8/10"
            trend={-2}
            color="yellow"
          />
        </motion.div>
      </motion.div>

      {/* Main Content */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Map Section */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <div className="bg-slate-800/30 backdrop-blur-xl border border-slate-700/30 rounded-xl overflow-hidden">
            <InteractiveMap />
          </div>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          variants={itemVariants}
          className="space-y-4"
        >
          <div className="glass-dark p-6 rounded-lg">
            <h3 className="text-white font-semibold mb-4">Quick Stats</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Slum Coverage</span>
                <span className="text-cyan-400 font-bold">542.5 km²</span>
              </div>
              <div className="w-full bg-slate-700/30 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '68%' }}
                  transition={{ duration: 1 }}
                  className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full"
                />
              </div>
              <div className="flex justify-between items-center pt-2">
                <span className="text-slate-400">Infrastructure Score</span>
                <span className="text-green-400 font-bold">58%</span>
              </div>
              <div className="w-full bg-slate-700/30 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '58%' }}
                  transition={{ duration: 1, delay: 0.2 }}
                  className="bg-gradient-to-r from-green-500 to-emerald-500 h-full rounded-full"
                />
              </div>
            </div>
          </div>

          <div className="glass-dark p-6 rounded-lg">
            <h3 className="text-white font-semibold mb-4">Region Overview</h3>
            <div className="space-y-2 text-sm max-h-64 overflow-y-auto">
              {regions.slice(0, 5).map((region, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex justify-between p-2 hover:bg-slate-700/30 rounded cursor-pointer transition-all"
                >
                  <span className="text-slate-300">{region.name || `Region ${idx + 1}`}</span>
                  <span className="text-cyan-400 text-xs font-bold">{(region.area_km2 || 2.5).toFixed(1)} km²</span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Analytics Chart */}
      <motion.div
        variants={itemVariants}
        initial="hidden"
        animate="visible"
        transition={{ delay: 0.4 }}
        className="mt-8"
      >
        <AnalyticsChart />
      </motion.div>
    </div>
  )
}

export default Dashboard
