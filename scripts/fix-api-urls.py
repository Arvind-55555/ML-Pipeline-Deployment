#!/usr/bin/env python3
"""
Script to update all HTML files to use API_BASE configuration
"""

import os
import re
from pathlib import Path

UI_DIR = Path(__file__).parent.parent / "ui"


def update_html_file(file_path):
    """Update a single HTML file with API configuration"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    changes = []

    # Add API config script if not present
    if "api-config.js" not in content:
        if "</head>" in content:
            content = content.replace(
                "</head>", '<script src="api-config.js"></script>\n    </head>'
            )
            changes.append("Added api-config.js script")

    # Add API_BASE declaration if not present
    if "const API_BASE" not in content and "window.API_BASE" not in content:
        # Find the first script tag and add API_BASE after it
        script_match = re.search(r"(<script[^>]*>)", content)
        if script_match:
            insert_pos = script_match.end()
            api_base_decl = (
                "\n        const API_BASE = window.API_BASE || '';\n        "
            )
            content = content[:insert_pos] + api_base_decl + content[insert_pos:]
            changes.append("Added API_BASE declaration")

    # Replace hardcoded API URLs
    patterns = [
        (r"fetch\('/", r"fetch(`${API_BASE}/"),
        (r'fetch\("/', r"fetch(`${API_BASE}/"),
        (r"fetch\('http://127\.0\.0\.1:8000/", r"fetch(`${API_BASE}/"),
        (r'fetch\("http://127\.0\.0\.1:8000/', r"fetch(`${API_BASE}/"),
        (r"'/health'", r"`${API_BASE}/health`"),
        (r'"/health"', r"`${API_BASE}/health`"),
    ]

    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f"Replaced {pattern}")

    # Fix template literals (ensure backticks)
    content = re.sub(r"fetch\(\`\$\{API_BASE\}/", r"fetch(`${API_BASE}/", content)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return changes
    return []


def main():
    """Main function"""
    html_files = list(UI_DIR.glob("*.html"))

    print(f"Found {len(html_files)} HTML files")

    for html_file in html_files:
        print(f"\nProcessing: {html_file.name}")
        changes = update_html_file(html_file)
        if changes:
            print("  ✅ Updated:")
            for change in changes:
                print(f"     - {change}")
        else:
            print("  ✓ No changes needed")


if __name__ == "__main__":
    main()
