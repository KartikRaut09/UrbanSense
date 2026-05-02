module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0f172a',
        secondary: '#1e293b',
        accent: '#06b6d4',
        danger: '#ef4444',
        warning: '#f59e0b',
        success: '#10b981',
        'slum-overlay': 'rgba(239, 68, 68, 0.3)',
        'fire-risk': 'rgba(239, 68, 68, 0.6)',
        'flood-risk': 'rgba(59, 130, 246, 0.6)',
        'high-risk': '#a71930',
        'medium-risk': '#f59e0b',
        'low-risk': '#10b981',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in': 'slide-in 0.5s ease-out',
        'fade-in': 'fade-in 0.3s ease-in',
        'scale-in': 'scale-in 0.4s ease-out',
        'bounce-slow': 'bounce 2s ease-in-out infinite',
        'drift': 'drift 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 0 0 rgba(6, 182, 212, 0.7)' },
          '50%': { opacity: '0.8', boxShadow: '0 0 0 10px rgba(6, 182, 212, 0)' },
        },
        'slide-in': {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'scale-in': {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        'drift': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '0% 0%' },
          '100%': { backgroundPosition: '100% 100%' },
        },
      },
      backdropFilter: {
        'glass': 'blur(10px)',
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        'glow': '0 0 20px rgba(6, 182, 212, 0.5)',
        'danger': '0 0 20px rgba(239, 68, 68, 0.5)',
      },
    },
  },
  plugins: [],
}
