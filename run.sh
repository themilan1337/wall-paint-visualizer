#!/bin/bash
# Quick start script

echo "=========================================="
echo "Wall Paint Visualizer API"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "⚠ Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
echo ""
echo "Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 5

# Check health
echo ""
echo "Checking API health..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "=========================================="
echo "✓ Services are running!"
echo "=========================================="
echo ""
echo "  🌐 API:      http://localhost:8000"
echo "  📚 API Docs: http://localhost:8000/docs"
echo "  💾 Redis:    localhost:6379"
echo ""
echo "View logs:     docker-compose logs -f"
echo "Stop services: docker-compose down"
echo ""
