.PHONY: help up down build restart logs test test-local clean

help:
	@echo "Available commands:"
	@echo "  make up          - Start all containers in background"
	@echo "  make down        - Stop and remove all containers"
	@echo "  make build       - Rebuild Docker images"
	@echo "  make restart     - Restart services"
	@echo "  make logs        - Follow container logs"
	@echo "  make test        - Run test suite inside Docker container"
	@echo "  make test-local  - Run test suite locally with pytest"
	@echo "  make clean       - Remove temporary artifacts and cache"

up:
	docker compose up --build -d

down:
	docker compose down

build:
	docker compose build

restart:
	docker compose restart

logs:
	docker compose logs -f

test:
	docker compose exec aml-app pytest tests/ -v

test-local:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
