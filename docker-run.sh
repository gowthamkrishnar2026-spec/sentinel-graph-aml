#!/usr/bin/env bash
# ==============================================================================
# Single-Command Launcher for AML Forensic & Fraud Detection System (Docker)
# ==============================================================================
set -e

echo "=================================================================="
echo " Anti-Money Laundering (AML) Network Forensics System"
echo " Researchers: Aakash R (23MIA1011) · Gowthamkrishna R (23MIA1087)"
echo "=================================================================="

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH."
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "🚀 Building and starting Docker containers..."
docker compose down --remove-orphans 2>/dev/null || true
docker compose up --build -d

echo ""
echo "⏳ Waiting for Neo4j and AML Engine to initialize..."
until curl -s http://localhost:8000/api/overview > /dev/null 2>&1; do
    echo "   ... still starting services (checking http://localhost:8000/api/overview) ..."
    sleep 3
done

echo ""
echo "=================================================================="
echo "✅ AML Forensic Platform is LIVE & READY!"
echo "=================================================================="
echo "🌐 Interactive Dashboard:    http://localhost:8000"
echo "📊 OpenAPI / Swagger Docs:    http://localhost:8000/docs"
echo "🗄️  Neo4j Browser Console:    http://localhost:7474"
echo "   (Credentials: neo4j / password123)"
echo ""
echo "To view live logs:             docker compose logs -f"
echo "To run automated tests:        docker compose exec aml-app pytest tests/ -v"
echo "To stop containers:            docker compose down"
echo "=================================================================="
