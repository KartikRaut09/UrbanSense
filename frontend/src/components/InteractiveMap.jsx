import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '../stores/useStore'
import { Layers, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'

function InteractiveMap() {
  const { mapView, setMapView, layersVisible, toggleLayer } = useStore()
  const [mapContainer, setMapContainer] = useState(null)

  // Layer toggle controls
  const layers = [
    { key: 'slums', label: 'Slum Areas', color: 'text-red-400' },
    { key: 'fireRisk', label: 'Fire Risk', color: 'text-orange-400' },
    { key: 'floodRisk', label: 'Flood Risk', color: 'text-blue-400' },
    { key: 'roads', label: 'Roads', color: 'text-amber-400' },
    { key: 'accessibility', label: 'Accessibility', color: 'text-green-400' },
  ]

  const handleZoom = (direction) => {
    const newZoom = direction === 'in'
      ? Math.min(mapView.zoom + 1, 20)
      : Math.max(mapView.zoom - 1, 1)
    setMapView({ zoom: newZoom })
  }

  const handleReset = () => {
    setMapView({
      latitude: 28.7041,
      longitude: 77.1025,
      zoom: 11
    })
  }

  return (
    <div className="relative h-96 bg-slate-900 rounded-lg overflow-hidden group">
      {/* Mock map canvas */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Grid pattern background */}
        <div className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `linear-gradient(0deg, transparent 24%, rgba(6, 182, 212, 0.05) 25%, rgba(6, 182, 212, 0.05) 26%, transparent 27%, transparent 74%, rgba(6, 182, 212, 0.05) 75%, rgba(6, 182, 212, 0.05) 76%, transparent 77%, transparent),
                            linear-gradient(90deg, transparent 24%, rgba(6, 182, 212, 0.05) 25%, rgba(6, 182, 212, 0.05) 26%, transparent 27%, transparent 74%, rgba(6, 182, 212, 0.05) 75%, rgba(6, 182, 212, 0.05) 76%, transparent 77%, transparent)`,
            backgroundSize: '50px 50px'
          }}
        />

        {/* Map content center */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="absolute inset-0 flex items-center justify-center"
        >
          <div className="text-center">
            <div className="text-slate-400 text-sm mb-2">
              {mapView.latitude.toFixed(4)}, {mapView.longitude.toFixed(4)}
            </div>
            <div className="text-cyan-400 font-semibold">Zoom: {mapView.zoom}</div>
          </div>
        </motion.div>

        {/* Simulated data points */}
        {layersVisible.slums && (
          <>
            {[...Array(8)].map((_, i) => (
              <motion.div
                key={`slum-${i}`}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.7 }}
                transition={{ delay: 0.4 + i * 0.05 }}
                className="absolute w-8 h-8 bg-red-500/30 border-2 border-red-400 rounded-full hover:bg-red-500/60 hover:border-red-300 transition-all cursor-pointer"
                style={{
                  left: `${15 + i * 10}%`,
                  top: `${20 + (i % 3) * 15}%`
                }}
              >
                <motion.div
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="absolute inset-0 bg-red-400/20 rounded-full"
                />
              </motion.div>
            ))}
          </>
        )}

        {layersVisible.fireRisk && (
          <>
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={`fire-${i}`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.5 + i * 0.06 }}
                className="absolute w-6 h-6 bg-orange-500/40 border border-orange-400 rounded"
                style={{
                  left: `${10 + i * 18}%`,
                  top: `${35 + (i % 2) * 20}%`
                }}
              />
            ))}
          </>
        )}

        {layersVisible.floodRisk && (
          <>
            {[...Array(6)].map((_, i) => (
              <motion.div
                key={`flood-${i}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.6 }}
                transition={{ delay: 0.5 + i * 0.05 }}
                className="absolute w-5 h-5 bg-blue-500/40 border border-blue-400 rounded-full"
                style={{
                  left: `${20 + i * 12}%`,
                  top: `${60 + (i % 2) * 15}%`
                }}
              />
            ))}
          </>
        )}
      </div>

      {/* Top Controls */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur-md border border-slate-700/50 rounded-lg p-3 z-20"
      >
        <div className="flex items-center gap-2 text-slate-300 text-xs mb-2">
          <Layers size={14} />
          <span className="font-semibold">Layers</span>
        </div>
        <div className="space-y-2">
          {layers.map((layer) => (
            <label key={layer.key} className="flex items-center gap-2 cursor-pointer group">
              <input
                type="checkbox"
                checked={layersVisible[layer.key]}
                onChange={() => toggleLayer(layer.key)}
                className="w-3 h-3 rounded"
              />
              <span className={`text-xs ${layer.color} group-hover:font-semibold transition-all`}>
                {layer.label}
              </span>
            </label>
          ))}
        </div>
      </motion.div>

      {/* Right Controls */}
      <motion.div
        initial={{ opacity: 0, x: 10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="absolute right-4 top-4 flex flex-col gap-2 z-20"
      >
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => handleZoom('in')}
          className="bg-cyan-500/20 hover:bg-cyan-500/40 border border-cyan-500/50 text-cyan-400 p-2 rounded-lg transition-all"
        >
          <ZoomIn size={18} />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => handleZoom('out')}
          className="bg-cyan-500/20 hover:bg-cyan-500/40 border border-cyan-500/50 text-cyan-400 p-2 rounded-lg transition-all"
        >
          <ZoomOut size={18} />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={handleReset}
          className="bg-cyan-500/20 hover:bg-cyan-500/40 border border-cyan-500/50 text-cyan-400 p-2 rounded-lg transition-all"
        >
          <RotateCcw size={18} />
        </motion.button>
      </motion.div>

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-md border border-slate-700/50 rounded-lg p-3 z-20"
      >
        <div className="text-xs text-slate-300 space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span>High Risk Zones</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-orange-500" />
            <span>Fire Hazard</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full" />
            <span>Flood Zone</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default InteractiveMap
