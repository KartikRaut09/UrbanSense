import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '../stores/useStore'
import RecommendationsList from '../components/RecommendationsList'
import { Lightbulb, TrendingUp, AlertTriangle } from 'lucide-react'

function Insights() {
  const { fetchInsights, insightsData } = useStore()

  useEffect(() => {
    fetchInsights('REGION_001')
  }, [fetchInsights])

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
        <h1 className="text-4xl font-bold text-white mb-2">AI-Generated Insights</h1>
        <p className="text-slate-400">Machine learning powered recommendations and analysis</p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Real-time Metrics */}
        <motion.div variants={itemVariants} className="lg:col-span-3">
          <div className="glass-dark p-6 rounded-lg">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Lightbulb size={24} className="text-yellow-400" />
              Key Findings
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { icon: AlertTriangle, label: 'Critical Issues', value: 3 },
                { icon: TrendingUp, label: 'Growth Areas', value: 5 },
                { icon: Lightbulb, label: 'Opportunities', value: 8 }
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  whileHover={{ translateY: -4 }}
                  className="bg-slate-700/30 p-4 rounded-lg border border-slate-600/30"
                >
                  <div className="flex items-center gap-3 mb-2">
                    <item.icon className="text-cyan-400" size={20} />
                    <span className="text-slate-400 text-sm">{item.label}</span>
                  </div>
                  <p className="text-2xl font-bold text-white">{item.value}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Recommendations */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <div className="glass-dark p-6 rounded-lg">
            <h2 className="text-xl font-semibold text-white mb-4">AI Recommendations</h2>
            <RecommendationsList
              recommendations={insightsData?.recommendations || [
                {
                  id: 'REC_001',
                  title: 'Emergency Road Development',
                  description: 'Develop primary emergency access road',
                  priority: 'Critical',
                  estimated_impact: 0.85,
                  implementation_cost: '$500K-$750K'
                },
                {
                  id: 'REC_002',
                  title: 'Fire Safety Infrastructure',
                  description: 'Install fire hydrant network and water tanks',
                  priority: 'High',
                  estimated_impact: 0.75,
                  implementation_cost: '$200K-$350K'
                },
                {
                  id: 'REC_003',
                  title: 'Drainage System Upgrade',
                  description: 'Improve stormwater drainage to reduce flood risk',
                  priority: 'High',
                  estimated_impact: 0.65,
                  implementation_cost: '$300K-$500K'
                }
              ]}
            />
          </div>
        </motion.div>

        {/* Prediction Summary */}
        <motion.div variants={itemVariants} className="lg:col-span-1">
          <div className="glass-dark p-6 rounded-lg h-full">
            <h3 className="text-white font-semibold mb-4">5-Year Forecast</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-400">Area Growth</span>
                  <span className="text-cyan-400 font-bold">+25%</span>
                </div>
                <div className="w-full bg-slate-700/30 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: '25%' }}
                    transition={{ duration: 1 }}
                    className="bg-gradient-to-r from-orange-500 to-red-500 h-full rounded-full"
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-400">Population</span>
                  <span className="text-cyan-400 font-bold">+32%</span>
                </div>
                <div className="w-full bg-slate-700/30 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: '32%' }}
                    transition={{ duration: 1, delay: 0.2 }}
                    className="bg-gradient-to-r from-orange-500 to-red-500 h-full rounded-full"
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-400">Risk Index</span>
                  <span className="text-cyan-400 font-bold">+18%</span>
                </div>
                <div className="w-full bg-slate-700/30 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: '18%' }}
                    transition={{ duration: 1, delay: 0.4 }}
                    className="bg-gradient-to-r from-orange-500 to-red-500 h-full rounded-full"
                  />
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default Insights
