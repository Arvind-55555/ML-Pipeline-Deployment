# 🔧 Fix: "No space left on device" Error

## The Problem

GitHub Actions runners sometimes run out of disk space when installing packages. However, **for GitHub Pages deployment, we don't need to install any Python packages!** We're just copying static HTML/CSS/JS files.

## The Solution

I've updated the workflow to:

1. ✅ **Clean up disk space** before building
2. ✅ **Skip Python package installation** (not needed for static files)
3. ✅ **Only copy necessary files** (HTML, JS, CSS)

## What Changed

The workflow now:
- Removes unnecessary tools to free up space
- Only copies UI files (no `pip install` needed)
- Uses minimal disk space

## If You Still Get Errors

### Option 1: Use the Minimal Workflow

Rename the minimal workflow:
```bash
mv .github/workflows/deploy-pages-minimal.yml .github/workflows/deploy-pages.yml
```

### Option 2: Manual Cleanup

If the workflow still fails, you can manually clean up in the workflow:

```yaml
- name: Clean up disk space
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /opt/ghc
    sudo rm -rf "$AGENT_TOOLSDIRECTORY"
    df -h
```

## Why This Happens

GitHub Actions runners have limited disk space. When workflows try to install large packages (like ML libraries), they can run out of space. But for static site deployment, we don't need any of that!

## Verification

After the fix, the workflow should:
- ✅ Complete in under 1 minute
- ✅ Use minimal disk space
- ✅ Successfully deploy to GitHub Pages

---

**Note:** The workflow has been updated. Just push the changes and it should work!

