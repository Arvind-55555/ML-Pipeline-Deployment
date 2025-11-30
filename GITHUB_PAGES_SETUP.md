# 🚀 GitHub Pages Deployment Guide

This guide will help you deploy the ML Pipeline UI to GitHub Pages.

## 📋 Prerequisites

1. A GitHub account
2. A GitHub repository for this project
3. GitHub Pages enabled (automatic with the workflow)

## 🔧 Setup Steps

### Step 1: Push Code to GitHub

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: ML Pipeline Deployment Platform"

# Add your GitHub repository as remote (replace with your repo URL)
git remote add origin https://github.com/Arvind-55555/ML-Pipeline-Deployment.git

# Push to main branch
git branch -M main
git push -u origin main
```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings**
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. The workflow will automatically deploy on push to main/master

### Step 3: Access Your Deployed Site

After the workflow runs (usually takes 1-2 minutes), your site will be available at:

```
https://github.com/Arvind-55555/ML-Pipeline-Deployment
```

For example: `https://johndoe.github.io/ml-pipe-deploy/`

## 🔗 API Backend Deployment

The UI on GitHub Pages is static, but it needs a backend API. You have several options:

### Option 1: Deploy API Separately (Recommended)

Deploy the FastAPI backend to one of these services:

#### Railway (Easiest)
1. Go to https://railway.app
2. Create new project
3. Connect your GitHub repo
4. Add `run_unified_server.py` as the start command
5. Get the deployment URL
6. Update `api-config.js` in the UI with the API URL

#### Render
1. Go to https://render.com
2. Create new Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python run_unified_server.py`
6. Get the deployment URL

#### Fly.io
1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Follow the prompts
4. Get the deployment URL

### Option 2: Update API Configuration

After deploying your API, update the API URL in the GitHub Pages site:

1. Edit `.github/workflows/deploy-pages.yml`
2. Update the `API_BASE_URL` variable:
   ```yaml
   API_BASE_URL: 'https://your-api.railway.app'
   ```
3. Push the changes
4. The workflow will redeploy with the new API URL

### Option 3: Use Environment Variable

You can also set the API URL as a GitHub secret:

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Add a new secret: `API_BASE_URL` with your API URL
3. Update the workflow to use: `${{ secrets.API_BASE_URL }}`

## 📝 Workflow Files

The repository includes two workflow files:

1. **`.github/workflows/deploy-pages.yml`** - Deploys UI to GitHub Pages
2. **`.github/workflows/deploy-api.yml`** - Template for API deployment (configure as needed)

## ✅ Verification

After deployment:

1. Check the **Actions** tab in your GitHub repository
2. Verify the workflow completed successfully
3. Visit your GitHub Pages URL
4. Test the UI - it should load and connect to your API

## 🔄 Updating the Site

Every time you push to the `main` or `master` branch, the site will automatically redeploy.

## 🐛 Troubleshooting

### Site not loading
- Check the **Actions** tab for workflow errors
- Verify GitHub Pages is enabled in Settings
- Wait a few minutes for DNS propagation

### API not connecting
- Verify your API is deployed and accessible
- Check the API URL in `api-config.js`
- Check browser console for CORS errors
- Ensure your API allows CORS from your GitHub Pages domain

### Workflow failing
- Check that all UI files are in the `ui/` directory
- Verify file permissions
- Check workflow logs in the Actions tab

## 📚 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

---

**Your deployed site will be available at:**
`https://github.com/Arvind-55555/ML-Pipeline-Deployment`

