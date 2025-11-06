.PHONY: help build up down restart train logs clean test frontend backend etl

# Default target
help:
	@echo "UrbanSim WM - Makefile Commands"
	@echo "================================"
	@echo "  make build       - Build all Docker containers"
	@echo "  make up          - Start all services (backend + frontend)"
	@echo "  make down        - Stop all services"
	@echo "  make restart     - Restart all services"
	@echo "  make train       - Run training service"
	@echo "  make logs        - Follow logs from all services"
	@echo "  make clean       - Remove containers, networks, and volumes"
	@echo "  make test        - Run tests (TODO)"
	@echo "  make frontend    - Build and start frontend only"
	@echo "  make backend     - Build and start backend only"
	@echo "  make etl         - Run ETL data fetchers"
	@echo "  make shell-be    - Open shell in backend container"
	@echo "  make shell-fe    - Open shell in frontend container"
	@echo ""

# Build all containers
build:
	@echo "Building all containers..."
	docker-compose build

# Start all services
up:
	@echo "Starting UrbanSim WM services..."
	docker-compose up -d backend frontend
	@echo ""
	@echo "Services started!"
	@echo "  - Frontend: http://localhost:3000"
	@echo "  - Backend API: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"
	@echo ""

# Stop all services
down:
	@echo "Stopping all services..."
	docker-compose down

# Restart all services
restart: down up

# Run training
train:
	@echo "Running training service..."
	docker-compose --profile training run --rm training

# View logs
logs:
	docker-compose logs -f

# View backend logs only
logs-backend:
	docker-compose logs -f backend

# View frontend logs only
logs-frontend:
	docker-compose logs -f frontend

# Clean up everything
clean:
	@echo "Cleaning up containers, networks, and volumes..."
	docker-compose down -v --remove-orphans
	@echo "Removing dangling images..."
	docker image prune -f
	@echo "Clean complete!"

# Run tests (TODO: implement tests)
test:
	@echo "Running tests..."
	@echo "TODO: Implement test suite"
	# docker-compose run --rm backend pytest
	# docker-compose run --rm frontend npm test

# Frontend only
frontend:
	docker-compose up -d frontend

# Backend only
backend:
	docker-compose up -d backend

# Run ETL data fetchers

# How to use:
# Fetch and write PM2.5 mean:
# make etl-openaq CITY="Lahore" HOURS=24
# Produces etl/processed_data/pm25_lahore.json

etl:
	@echo "Running ETL data fetchers..."
	@echo "Fetching OpenAQ data..."
	python etl/fetch_openaq.py --city "${CITY}" --hours ${HOURS}
	@echo "Fetching mobility data..."
	python etl/fetch_mobility.py
	@echo "Fetching energy data..."
	python etl/fetch_energy.py

etl-openaq:
	@echo "Fetching OpenAQ PM2.5 mean..."
	python etl/fetch_openaq.py --city "${CITY}" --hours ${HOURS}

# Open shell in backend container
shell-be:
	docker-compose exec backend /bin/bash

# Open shell in frontend container
shell-fe:
	docker-compose exec frontend /bin/sh

# Development: rebuild and restart
dev: build up logs

# Check service status
status:
	docker-compose ps

# Install dependencies locally (for development without Docker)
install-backend:
	cd backend && pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

install-training:
	cd training && pip install -r requirements.txt

install-etl:
	cd etl && pip install -r requirements.txt

install: install-backend install-frontend install-training install-etl

