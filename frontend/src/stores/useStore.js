import { create } from 'zustand'
import api from '../services/api'

export const useStore = create((set, get) => ({
  // State
  selectedRegion: null,
  regions: [],
  analysisData: null,
  riskData: null,
  insightsData: null,
  loading: false,
  error: null,
  mapView: {
    latitude: 28.7041,
    longitude: 77.1025,
    zoom: 11,
  },
  layersVisible: {
    slums: true,
    fireRisk: false,
    floodRisk: false,
    roads: true,
    accessibility: false,
  },
  timeframe: 'current',

  // Actions
  setSelectedRegion: (region) => set({ selectedRegion: region }),
  setRegions: (regions) => set({ regions }),
  setAnalysisData: (data) => set({ analysisData: data }),
  setRiskData: (data) => set({ riskData: data }),
  setInsightsData: (data) => set({ insightsData: data }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setMapView: (view) => set((state) => ({
    mapView: { ...state.mapView, ...view }
  })),
  toggleLayer: (layerName) => set((state) => ({
    layersVisible: {
      ...state.layersVisible,
      [layerName]: !state.layersVisible[layerName]
    }
  })),
  setTimeframe: (timeframe) => set({ timeframe }),

  // API Calls
  fetchRegions: async () => {
    set({ loading: true })
    try {
      const response = await api.get('/data/regions')
      set({ regions: response.data.regions, error: null, loading: false })
    } catch (error) {
      set({ error: error.message, loading: false })
    }
  },

  fetchAnalysis: async (regionId) => {
    set({ loading: true })
    try {
      const response = await api.post('/analysis/region', {
        region_id: regionId,
        latitude: 28.7041,
        longitude: 77.1025,
        zoom_level: 11
      })
      set({ analysisData: response.data, error: null, loading: false })
    } catch (error) {
      set({ error: error.message, loading: false })
    }
  },

  fetchRiskAssessment: async (regionId) => {
    set({ loading: true })
    try {
      const response = await api.post('/risk/assess', {
        latitude: 28.7041,
        longitude: 77.1025,
        area_name: regionId
      })
      set({ riskData: response.data, error: null, loading: false })
    } catch (error) {
      set({ error: error.message, loading: false })
    }
  },

  fetchInsights: async (regionId) => {
    set({ loading: true })
    try {
      const response = await api.get(`/insights/ai?region_id=${regionId}`)
      set({ insightsData: response.data, error: null, loading: false })
    } catch (error) {
      set({ error: error.message, loading: false })
    }
  },

  clearError: () => set({ error: null }),
}))
