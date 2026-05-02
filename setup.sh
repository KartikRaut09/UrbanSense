#!/bin/bash
# UrbanSense Quick Start Script

set -e

echo "================================"
echo "UrbanSense Initialization Script"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Check Node
echo ""
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"

# Check Git
echo ""
echo "Checking Git..."
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git found: $(git --version)${NC}"

# Backend Setup
echo ""
echo -e "${YELLOW}Setting up Backend...${NC}"
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

echo "Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null

if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${YELLOW}Note: Update .env file with your configuration${NC}"
fi

echo -e "${GREEN}✓ Backend setup complete${NC}"

# Frontend Setup
echo ""
echo -e "${YELLOW}Setting up Frontend...${NC}"
cd ../frontend

echo "Installing dependencies..."
npm install > /dev/null

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "VITE_API_URL=http://localhost:8000/api/v1" > .env
fi

echo -e "${GREEN}✓ Frontend setup complete${NC}"

# Summary
echo ""
echo "================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Update backend/.env with your configuration"
echo "2. Start backend:   cd backend && source venv/bin/activate && python main.py"
echo "3. Start frontend:  cd frontend && npm run dev"
echo ""
echo "Services will be available at:"
echo "  - Frontend:  http://localhost:5173"
echo "  - Backend:   http://localhost:8000"
echo "  - API Docs:  http://localhost:8000/docs"
echo ""
