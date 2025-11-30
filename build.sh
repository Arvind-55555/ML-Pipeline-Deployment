#!/bin/bash
# Build script for ML Pipeline Deployment package

set -e

echo "🔨 Building ML Pipeline Deployment package..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install build tools
echo "📦 Installing build tools..."
pip install --upgrade build twine

# Build package
echo "🏗️  Building package..."
python -m build

# Check package
echo "✅ Checking package..."
python -m twine check dist/*

echo ""
echo "✅ Build complete!"
echo "📦 Package files:"
ls -lh dist/

echo ""
echo "📋 Next steps:"
echo "  1. Test on Test PyPI:"
echo "     python -m twine upload --repository testpypi dist/*"
echo ""
echo "  2. Publish to PyPI:"
echo "     python -m twine upload dist/*"
echo ""

