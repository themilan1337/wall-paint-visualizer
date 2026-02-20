.PHONY: help install dev build up down logs test clean

help:
	@echo "Wall Paint Visualizer - Available commands:"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Run in development mode"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start Docker containers"
	@echo "  make down       - Stop Docker containers"
	@echo "  make logs       - View Docker logs"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean cache and build files"
	@echo ""

install:
	pip install -r requirements.txt

dev:
	@echo "Starting in development mode..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

build:
	docker-compose build

up:
	docker-compose up -d
	@echo ""
	@echo "✓ Services started!"
	@echo "  API: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  Redis: localhost:6379"
	@echo ""
	@echo "View logs: make logs"

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	@echo "Running API health check..."
	curl -s http://localhost:8000/health | python -m json.tool

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .cache build dist
	@echo "✓ Cleaned cache and build files"
