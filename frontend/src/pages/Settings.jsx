import React from 'react'
import { motion } from 'framer-motion'
import { Settings as SettingsIcon, Database, MapPin, Zap } from 'lucide-react'

function Settings() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  }

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-white mb-2">Settings</h1>
        <p className="text-slate-400">Configure your UrbanSense experience</p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >
        {/* Map Settings */}
        <motion.div variants={itemVariants} className="glass-dark p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <MapPin className="text-cyan-400" size={24} />
            <h2 className="text-xl font-semibold text-white">Map Preferences</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Default Map Style</p>
                <p className="text-slate-400 text-sm">Choose your preferred map visualization</p>
              </div>
              <select className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white">
                <option>Dark</option>
                <option>Light</option>
                <option>Satellite</option>
              </select>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Default Zoom Level</p>
                <p className="text-slate-400 text-sm">Initial map zoom when loading</p>
              </div>
              <input type="range" min="1" max="20" defaultValue="11" className="w-32" />
            </div>
          </div>
        </motion.div>

        {/* Data Settings */}
        <motion.div variants={itemVariants} className="glass-dark p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <Database className="text-cyan-400" size={24} />
            <h2 className="text-xl font-semibold text-white">Data Configuration</h2>
          </div>
          <div className="space-y-4">
            <label className="flex items-center gap-3">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-slate-300">Enable satellite imagery caching</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-slate-300">Auto-sync with Google Earth Engine</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-4 h-4" />
              <span className="text-slate-300">Real-time data streaming</span>
            </label>
          </div>
        </motion.div>

        {/* Performance Settings */}
        <motion.div variants={itemVariants} className="glass-dark p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="text-cyan-400" size={24} />
            <h2 className="text-xl font-semibold text-white">Performance</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">Animation Quality</p>
                <p className="text-slate-400 text-sm">Adjust animation smoothness</p>
              </div>
              <select className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white">
                <option>High</option>
                <option>Medium</option>
                <option>Low</option>
              </select>
            </div>
            <label className="flex items-center gap-3">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-slate-300">Enable GPU acceleration</span>
            </label>
          </div>
        </motion.div>

        {/* About */}
        <motion.div variants={itemVariants} className="glass-dark p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <SettingsIcon className="text-cyan-400" size={24} />
            <h2 className="text-xl font-semibold text-white">About UrbanSense</h2>
          </div>
          <div className="space-y-2 text-sm text-slate-400">
            <p><span className="text-slate-300">Version:</span> 1.0.0</p>
            <p><span className="text-slate-300">Backend:</span> FastAPI v0.104.1</p>
            <p><span className="text-slate-300">Frontend:</span> React 18.2.0 with Vite</p>
            <p><span className="text-slate-300">Last Updated:</span> Jan 10, 2024</p>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <div className="flex gap-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold py-3 rounded-lg"
          >
            Save Settings
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex-1 bg-slate-700 text-slate-300 font-semibold py-3 rounded-lg hover:bg-slate-600"
          >
            Reset to Default
          </motion.button>
        </div>
      </motion.div>
    </div>
  )
}

export default Settings
