# 📦 Publishing ML Pipeline Deployment Package

## Overview

This guide explains how to publish the ML Pipeline Deployment Platform as a Python package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on:
   - Test PyPI: https://test.pypi.org/account/register/
   - Production PyPI: https://pypi.org/account/register/

2. **Install Build Tools**:
   ```bash
   pip install build twine
   ```

## Step 1: Update Package Information

1. Edit `pyproject.toml` or `setup.py`:
   - Update `author` and `author_email`
   - Update `version` (use semantic versioning)
   - Update `url` and repository links

2. Update `src/__init__.py` with correct version

## Step 2: Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Verify the build
ls -la dist/
```

You should see:
- `ml-pipeline-deploy-1.0.0.tar.gz` (source distribution)
- `ml_pipeline_deploy-1.0.0-py3-none-any.whl` (wheel)

## Step 3: Test on Test PyPI

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ ml-pipeline-deploy
```

## Step 4: Publish to Production PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token (create at https://pypi.org/manage/account/token/)

## Step 5: Verify Installation

```bash
# Install from PyPI
pip install ml-pipeline-deploy

# Test the package
ml-pipeline info
python -m ml_pipeline_deploy.run_unified_server
```

## Installation Options

### Basic Installation
```bash
pip install ml-pipeline-deploy
```

### With Full Dependencies
```bash
pip install ml-pipeline-deploy[full]
```

### Development Mode
```bash
pip install -e ".[dev]"
```

## Package Structure

```
ml-pipeline-deploy/
├── src/
│   ├── core/          # Pipeline orchestration
│   ├── stages/        # Pipeline stages
│   ├── utils/         # Utilities
│   └── deployment/    # Deployment tools
├── examples/          # 5 enterprise examples
├── ui/                # Web UIs
├── configs/           # Configuration files
└── run_unified_server.py
```

## Version Management

Update version in:
1. `pyproject.toml` - `version = "X.Y.Z"`
2. `setup.py` - `version="X.Y.Z"`
3. `src/__init__.py` - `__version__ = "X.Y.Z"`

Use semantic versioning:
- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install build twine
      - run: python -m build
      - run: python -m twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

## Troubleshooting

### Error: Package already exists
- Update version number
- Or use `--skip-existing` flag

### Error: Invalid credentials
- Create new API token at https://pypi.org/manage/account/token/
- Use `__token__` as username

### Error: Package too large
- Remove unnecessary files from `MANIFEST.in`
- Use `.gitignore` to exclude large files

## Post-Publication

1. **Create GitHub Release**: Tag the version
2. **Update Documentation**: Update README with installation instructions
3. **Announce**: Share on social media, forums, etc.

## Installation from GitHub

Users can also install directly from GitHub:

```bash
pip install git+https://github.com/Arvind-55555/ML-Pipeline-Deployment.git
```

## Next Steps

After publishing:
1. ✅ Package is available on PyPI
2. ✅ Users can `pip install ml-pipeline-deploy`
3. ✅ Create documentation site
4. ✅ Add badges to README
5. ✅ Set up CI/CD for automated releases

---

**Ready to publish?** Follow the steps above and your package will be available worldwide! 🚀

