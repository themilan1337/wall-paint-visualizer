# Wall Paint Visualizer API

AI-powered backend API for visualizing wall paint colors in room photos using semantic segmentation.

## Features

- 🎨 **2000+ Paint Colors** - Search and browse extensive color database
- 🤖 **AI Wall Detection** - SegFormer-based semantic segmentation
- ⚡ **Redis Caching** - Ultra-fast processing with mask caching
- 🚀 **FastAPI** - Modern, high-performance async API
- 🐳 **Docker Ready** - Production-ready containerization
- 📊 **RESTful API** - Clean, well-documented endpoints

## Quick Start with Docker

```bash
# Clone repository
git clone <repo-url>
cd wall-paint-visualizer

# Start services
docker-compose up -d

# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start Redis (or set REDIS_ENABLED=false in .env)
docker run -d -p 6379:6379 redis:7-alpine

# Run application
python -m app.main
```

## API Endpoints

### Colors
- `GET /api/colors/search?q=F00&limit=20` - Search colors by code
- `GET /api/colors/{code}` - Get specific color
- `GET /api/colors/` - List all color codes

### Images
- `POST /api/images/upload` - Upload image
- `GET /api/images/list` - List uploaded images
- `GET /api/images/{filename}` - Get image file
- `GET /api/images/edited/{filename}` - Get processed image

### Processing
- `POST /api/process/` - Process image with wall color change

Example request:
```json
{
  "image": "room.jpg",
  "color": "220,180,170"
}
```

### Health
- `GET /health` - Health check with system status
- `GET /` - API information

## Architecture

```
wall-paint-visualizer/
├── app/
│   ├── api/routes/       # API endpoints
│   ├── core/             # Config, cache, utilities
│   ├── services/         # Business logic
│   └── models/           # Data schemas
├── data/                 # Color database
├── public/               # Static files
│   ├── images/          # Uploaded images
│   ├── edited/          # Processed images
│   └── patterns/        # Pattern textures
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Performance Optimizations

1. **Redis Caching**: Wall masks are cached to avoid re-running AI model
2. **Lazy Loading**: AI model loads only when needed
3. **Async Processing**: FastAPI async endpoints for better concurrency
4. **Guided Filter**: Edge refinement for high-quality results
5. **Smart Scaling**: Dynamic image scaling for optimal quality/speed

## Environment Variables

See `.env.example` for all available configuration options.

Key settings:
- `REDIS_ENABLED`: Enable/disable Redis caching
- `MODEL_NAME`: SegFormer model to use
- `DEVICE`: cpu/cuda/mps for model inference
- `CACHE_TTL`: Cache expiration time

## License

MIT
