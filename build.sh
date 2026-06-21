#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Building frontend..."
cd nba-frontend
npm install
CI=false npm run build
cd ..

echo "Copying frontend build to backend..."
rm -rf sports-backend/build
cp -r nba-frontend/build sports-backend/build

echo "Installing backend dependencies..."
cd sports-backend
pip install -r requirements.txt
