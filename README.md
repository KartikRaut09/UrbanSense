# UrbanSense - Intelligent Slum Mapping & Urban Risk Intelligence System

An AI-powered geospatial platform for detecting, analyzing, and managing urban slum risks through satellite imagery processing, machine learning, and advanced data visualization.

## 🎯 Project Vision

UrbanSense transforms raw satellite imagery into actionable urban intelligence, enabling:
- **Automated slum detection** using deep learning segmentation
- **Multi-dimensional risk assessment** (fire, flood, accessibility)
- **Growth prediction** using LSTM time-series modeling
- **Intelligence-driven recommendations** for urban planning
- **Real-time interactive visualization** with advanced geospatial mapping

## 🏗️ System Architecture

```
┌─────────────────────────────┐
│  Satellite + GIS Data       │
│  (Sentinel-2, Google EE)    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Data Processing Layer      │
│  (Rasterio, GeoPandas)      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  AI Models Pipeline         │
│  • U-Net Segmentation       │
│  • Feature Extraction       │
│  • Risk Prediction (XGBoost)│
│  • Growth (LSTM)            │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Risk Intelligence Engine   │
│  • Rule-based Analysis      │
│  • Recommendations          │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  FastAPI Backend            │
│  RESTful APIs               │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  React Frontend             │
│  (Vite + Animations)        │
└─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 13+ with PostGIS
- Redis 6+

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python db.py

# Run development server
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Services will be available at:
# - Frontend: http://localhost:5173
# - Backend: http://localhost:8000
# - Database: localhost:5432
# - Redis: localhost:6379
```

## 📚 API Documentation

### Core Endpoints

#### Analysis Routes
```
POST /api/v1/analysis/region
  - Perform comprehensive urban analysis
  - Input: region_id, latitude, longitude, zoom_level
  - Output: area_metrics, infrastructure_score, livability_index

GET /api/v1/analysis/dashboard/{region_id}
  - Get dashboard data for region

GET /api/v1/analysis/statistics
  - Platform-wide statistics
```

#### Segmentation Routes
```
POST /api/v1/segmentation/detect
  - Detect slums in satellite imagery
  - Input: image_path, confidence_threshold
  - Output: segmentation_map, confidence_scores, slum_area_percentage

POST /api/v1/segmentation/upload
  - Upload satellite image for processing
```

#### Risk Analysis Routes
```
POST /api/v1/risk/assess
  - Assess multiple risk factors
  - Input: latitude, longitude, area_name
  - Output: fire_risk, flood_risk, accessibility_risk, recommendations

GET /api/v1/risk/heatmap/{region_id}
  - Get risk heatmap for region

GET /api/v1/risk/historical/{area_id}
  - Get historical risk trends
```

#### Insights Routes
```
GET /api/v1/insights/ai?region_id={region_id}
  - Get AI-generated insights
  - Output: recommendations, confidence_score

GET /api/v1/insights/predictions?region_id={region_id}&forecast_years={years}
  - Get slum growth predictions

GET /api/v1/insights/trends?region_id={region_id}
  - Get urban development trends
```

#### Data Routes
```
GET /api/v1/data/sources
  - List available data sources

GET /api/v1/data/regions
  - List analyzed regions

GET /api/v1/data/region/{region_id}
  - Get detailed region data

POST /api/v1/data/upload
  - Upload new region data

GET /api/v1/data/export/{region_id}
  - Export region data in various formats
```

## 🤖 AI/ML Pipeline

### Stage 1: Slum Segmentation
- **Model**: U-Net / DeepLabV3+
- **Input**: Multi-spectral satellite imagery (RGB + infrared)
- **Output**: Pixel-wise slum region masks
- **Framework**: PyTorch

### Stage 2: Feature Extraction
From detected areas, extract:
- Building density
- Area compactness
- Road connectivity
- Roof material patterns
- Vegetation indices (NDVI)

### Stage 3: Graph Analysis
- Build spatial graph of buildings
- Calculate clustering coefficients
- Determine connectivity metrics
- Tool: NetworkX

### Stage 4: Risk Prediction
- **Models**: XGBoost, LightGBM
- **Predict**:
  - Fire risk (density + materials + accessibility)
  - Flood risk (elevation + proximity to water + drainage)
  - Accessibility risk (hospital distance + road density)

### Stage 5: Growth Prediction
- **Model**: CNN + LSTM
- **Input**: Time-series satellite imagery
- **Output**: Future expansion zones with confidence intervals

## 🎨 Frontend Features

### Interactive Dashboard
- Real-time map visualization with Deck.gl
- Multi-layer overlay system
- Dynamic risk heatmaps
- Zone highlighting with animations

### Advanced UI Components
- Glass morphism design patterns
- Smooth Framer Motion animations
- Responsive grid layouts
- Real-time data updates

### Key Pages
1. **Dashboard** - Overview and KPIs
2. **Analysis** - Detailed area metrics
3. **Risk Assessment** - Risk scoring and heatmaps
4. **Insights** - AI recommendations
5. **Settings** - Configuration options

## 📊 Tech Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **ML/DL**: PyTorch, TensorFlow, scikit-learn
- **Geospatial**: Rasterio, GeoPandas, Google Earth Engine
- **Database**: PostgreSQL + PostGIS
- **Cache**: Redis
- **Task Queue**: Celery

### Frontend
- **UI Framework**: React 18.2.0
- **Build Tool**: Vite 5.0
- **Styling**: Tailwind CSS 3.3.6
- **Animations**: Framer Motion 10.16
- **State**: Zustand 4.4.1
- **Mapping**: Deck.gl 13.1.0, Mapbox GL
- **Charts**: Recharts 2.10.3

### Deployment
- **Backend**: AWS EC2 / Render
- **Frontend**: Vercel
- **MLOps**: Docker
- **CI/CD**: GitHub Actions

## 🔧 Configuration

### Environment Variables (.env)

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/urbansense
REDIS_URL=redis://localhost:6379/0

# API Configuration
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENVIRONMENT=development
DEBUG=True

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Google Earth Engine
GOOGLE_EARTH_ENGINE_KEY=path/to/gee-key.json

# Logging
LOG_LEVEL=INFO
```

## 📈 Evaluation Metrics

- **Segmentation**: IoU (Intersection over Union), Precision, Recall, F1-score
- **Risk Models**: Accuracy, ROC-AUC, RMSE
- **Growth Prediction**: MAE, RMSE on test set
- **Real-world**: Validation with ground truth data

## 🚢 Deployment

### Docker Deployment
```bash
# Build images
docker build -f Dockerfile.backend -t urbansense-backend .
docker build -f Dockerfile.frontend -t urbansense-frontend .

# Run containers
docker run -p 8000:8000 urbansense-backend
docker run -p 3000:3000 urbansense-frontend
```

### Kubernetes Deployment
- Use provided Helm charts
- Configure ingress for load balancing
- Setup persistent volumes for data

## 📝 Development Roadmap

### Phase 1 ✅
- Basic slum detection with U-Net
- Simple map UI
- Basic API endpoints

### Phase 2 🚀
- Risk analysis (fire + flood)
- Multi-layer visualization
- Database integration

### Phase 3 
- Multi-modal data integration
- Graph analysis
- Advanced feature extraction

### Phase 4
- Growth prediction model
- Time-series analysis
- Forecasting UI

### Phase 5
- AI recommendations engine
- LLM integration
- Policy insights
- Real-time alerts

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Sentinel-2 mission by ESA
- Google Earth Engine platform
- OpenStreetMap contributors
- Urban analytics research community

## 📞 Support

For issues, feature requests, or questions:
- GitHub Issues: [Create an issue](https://github.com/urbansense/issues)
- Email: support@urbansense.io
- Documentation: [https://docs.urbansense.io](https://docs.urbansense.io)

---

**Built with ❤️ for urban sustainability**
