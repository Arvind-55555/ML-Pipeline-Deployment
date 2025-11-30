#!/bin/bash
# Script to update all HTML files with API configuration

echo "Updating HTML files with API configuration..."

for file in ui/*.html; do
    if [ -f "$file" ]; then
        # Add API config script if not present
        if ! grep -q "api-config.js" "$file"; then
            sed -i '/<\/head>/i <script src="api-config.js"></script>' "$file"
        fi
        
        # Replace hardcoded API URLs with API_BASE variable
        sed -i "s|fetch('/|fetch(\`\${API_BASE}/|g" "$file"
        sed -i "s|fetch(\"/|fetch(\`\${API_BASE}/|g" "$file"
        sed -i "s|'/health'|\`\${API_BASE}/health\`|g" "$file"
        sed -i "s|\"/health\"|\`\${API_BASE}/health\`|g" "$file"
        
        echo "Updated: $file"
    fi
done

echo "✅ All HTML files updated!"

