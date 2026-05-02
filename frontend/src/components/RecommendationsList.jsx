import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ExternalLink, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react'

function RecommendationCard({ recommendation, index }) {
  const [expanded, setExpanded] = useState(false)

  const priorityColors = {
    Critical: 'border-l-4 border-red-500 bg-red-500/10 hover:bg-red-500/20',
    High: 'border-l-4 border-orange-500 bg-orange-500/10 hover:bg-orange-500/20',
    Medium: 'border-l-4 border-yellow-500 bg-yellow-500/10 hover:bg-yellow-500/20',
    Low: 'border-l-4 border-green-500 bg-green-500/10 hover:bg-green-500/20',
  }

  const priorityIcons = {
    Critical: <AlertCircle className="text-red-400" size={18} />,
    High: <AlertCircle className="text-orange-400" size={18} />,
    Medium: <TrendingUp className="text-yellow-400" size={18} />,
    Low: <CheckCircle className="text-green-400" size={18} />,
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      onClick={() => setExpanded(!expanded)}
      className={`rounded-lg p-4 cursor-pointer transition-all ${priorityColors[recommendation.priority]} border`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3 flex-1">
          <div className="mt-1">{priorityIcons[recommendation.priority]}</div>
          <div className="flex-1">
            <h4 className="text-white font-semibold mb-1">{recommendation.title}</h4>
            <p className="text-slate-300 text-sm">{recommendation.description}</p>

            <AnimatePresence>
              {expanded && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 space-y-2 text-xs"
                >
                  <div className="flex justify-between text-slate-400">
                    <span>Estimated Impact: <span className="text-cyan-400 font-semibold">{(recommendation.estimated_impact * 100).toFixed(0)}%</span></span>
                    <span>Cost: <span className="text-cyan-400 font-semibold">{recommendation.implementation_cost}</span></span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
        <motion.div
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <ExternalLink size={18} className="text-slate-400" />
        </motion.div>
      </div>
    </motion.div>
  )
}

function RecommendationsList({ recommendations = [] }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="space-y-3"
    >
      {recommendations.length > 0 ? (
        recommendations.map((rec, idx) => (
          <RecommendationCard key={idx} recommendation={rec} index={idx} />
        ))
      ) : (
        <div className="text-center py-8 text-slate-400">
          <p>No recommendations available</p>
        </div>
      )}
    </motion.div>
  )
}

export default RecommendationsList
