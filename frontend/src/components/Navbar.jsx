import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Menu, X, Home, BarChart3, AlertTriangle, Lightbulb, Settings, Globe } from 'lucide-react'

function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  const navItems = [
    { name: 'Dashboard', icon: Home, path: '/' },
    { name: 'Analysis', icon: BarChart3, path: '/analysis' },
    { name: 'Risk', icon: AlertTriangle, path: '/risk' },
    { name: 'Insights', icon: Lightbulb, path: '/insights' },
    { name: 'Settings', icon: Settings, path: '/settings' },
  ]

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 w-full bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 backdrop-blur-xl border-b border-cyan-500/20 z-50 h-16"
    >
      <div className="px-4 h-full flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 group">
          <motion.div
            whileHover={{ scale: 1.05, rotate: 5 }}
            className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center"
          >
            <Globe size={20} className="text-white" />
          </motion.div>
          <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            UrbanSense
          </span>
        </Link>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center gap-1">
          {navItems.map((item) => (
            <Link key={item.name} to={item.path}>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all duration-300"
              >
                <item.icon size={18} />
                <span className="text-sm font-medium">{item.name}</span>
              </motion.button>
            </Link>
          ))}
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden text-slate-300"
          onClick={() => setIsOpen(!isOpen)}
        >
          <motion.div
            animate={{ rotate: isOpen ? 90 : 0 }}
            transition={{ duration: 0.3 }}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </motion.div>
        </button>
      </div>

      {/* Mobile Menu */}
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: isOpen ? 1 : 0, height: isOpen ? 'auto' : 0 }}
        transition={{ duration: 0.3 }}
        className="md:hidden overflow-hidden"
      >
        <div className="bg-slate-800/50 border-t border-cyan-500/10 px-4 py-2 space-y-1">
          {navItems.map((item) => (
            <Link key={item.name} to={item.path}>
              <motion.button
                whileHover={{ translateX: 4 }}
                className="w-full flex items-center gap-2 px-4 py-2 rounded-lg text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all"
              >
                <item.icon size={18} />
                <span className="text-sm">{item.name}</span>
              </motion.button>
            </Link>
          ))}
        </div>
      </motion.div>
    </motion.nav>
  )
}

export default Navbar
