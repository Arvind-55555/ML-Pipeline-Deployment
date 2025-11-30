# 🚀 Quick Deployment Guide

## GitHub Pages Deployment (UI)

### 1. Push to GitHub

```bash
git add .
git commit -m "Setup GitHub Pages deployment"
git push origin main
```

### 2. Enable GitHub Pages

1. Go to your repository → **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. The workflow will automatically deploy

### 3. Your Site URL

After deployment (1-2 minutes), your site will be at:

```
https://github.com/Arvind-55555/ML-Pipeline-Deployment
```

## API Backend Deployment

The UI needs a backend API. Deploy to one of these:

### Railway (Recommended - Free tier available)

1. Sign up at https://railway.app
2. New Project → Deploy from GitHub
3. Select your repository
4. Add environment variable: `PORT=8000`
5. Deploy!

### Render (Free tier available)

1. Sign up at https://render.com
2. New Web Service
3. Connect GitHub repo
4. Build: `pip install -r requirements.txt`
5. Start: `python run_unified_server.py`

### Update API URL

After deploying your API:

1. Edit `.github/workflows/deploy-pages.yml`
2. Change: `API_BASE_URL: 'https://your-api.railway.app'`
3. Push changes
4. Site will redeploy with new API URL

## Quick Commands

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Initial commit"

# Add remote (replace with your repo)
git remote add origin https://github.com/Arvind-55555/ML-Pipeline-Deployment.git
git branch -M main
git push -u origin main
```

That's it! 🎉

