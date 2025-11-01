#!/bin/bash

set -e  # Stop on error

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Build completed successfully."
