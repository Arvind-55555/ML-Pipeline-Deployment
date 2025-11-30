# 🚀 GitHub Pages Deployment

## Quick Start

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Setup GitHub Pages"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Go to repository **Settings** → **Pages**
   - Source: Select **GitHub Actions**
   - Save

3. **Your Site URL:**
   ```
   https://github.com/Arvind-55555/ML-Pipeline-Deployment
   ```

## API Backend

The UI needs a backend API. Deploy to:

- **Railway:** https://railway.app (Free tier)
- **Render:** https://render.com (Free tier)
- **Fly.io:** https://fly.io (Free tier)

After deploying, update the API URL in `.github/workflows/deploy-pages.yml`

## Full Documentation

See `GITHUB_PAGES_SETUP.md` for detailed instructions.

