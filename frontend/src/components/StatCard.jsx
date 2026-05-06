import React from 'react'
import { motion } from 'framer-motion'
import { TrendingUp } from 'lucide-react'

function StatCard({ icon: Icon, label, value, trend, color = 'cyan' }) {
  const colorClasses = {
    cyan: 'from-cyan-500/20 to-blue-500/10 text-cyan-400',
    red: 'from-red-500/20 to-orange-500/10 text-red-400',
    yellow: 'from-yellow-500/20 to-orange-500/10 text-yellow-400',
    green: 'from-green-500/20 to-emerald-500/10 text-green-400',
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02, translateY: -4 }}
      className={`bg-gradient-to-br ${colorClasses[color]} backdrop-blur-xl border border-white/10 rounded-xl p-6 cursor-pointer group overflow-hidden relative`}
    >
      {/* Background effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
        animate={{ translateX: ['0%', '100%'] }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      <div className="relative z-10 flex items-start justify-between">
        <div className="flex-1">
          <p className="text-slate-400 text-sm font-medium mb-2">{label}</p>
          <p className="text-3xl font-bold text-white mb-2">{value}</p>
          {typeof trend === 'number' && (
            <div className="flex items-center gap-1 text-xs">
              <TrendingUp size={14} className={trend >= 0 ? 'text-green-400' : 'text-red-400'} />
              <span className={trend >= 0 ? 'text-green-400' : 'text-red-400'}>
                {trend >= 0 ? '+' : ''}{trend}%
              </span>
            </div>
          )}
        </div>
        <motion.div
          whileHover={{ rotate: 10, scale: 1.1 }}
          className={`p-3 rounded-lg bg-white/10 group-hover:bg-white/20 transition-all duration-300`}
        >
          <Icon size={24} />
        </motion.div>
      </div>
    </motion.div>
  )
}

export default StatCard
