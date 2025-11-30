"""
Command-line interface for ML Pipeline Deployment Platform
"""

import click
from pathlib import Path


@click.group()
@click.version_option(version="1.0.0")
def main():
    """ML Pipeline Deployment Platform - Enterprise ML Pipeline Examples"""
    pass


@main.command()
def info():
    """Show package information"""
    click.echo("=" * 70)
    click.echo("🚀 ML Pipeline Deployment Platform")
    click.echo("=" * 70)
    click.echo("\n📦 Version: 1.0.0")
    click.echo("📚 5 Enterprise ML Pipeline Examples")
    click.echo("🌐 GitHub: https://github.com/Arvind-55555/ML-Pipeline-Deployment")
    click.echo("\nAvailable Examples:")
    click.echo("  1. 📊 Customer Churn Prediction")
    click.echo("  2. 🚨 Real-time Fraud Detection")
    click.echo("  3. 🏥 Medical Image Analysis")
    click.echo("  4. 🛒 E-commerce Recommendation")
    click.echo("  5. 🚗 Autonomous Vehicle Perception")
    click.echo("\n" + "=" * 70)


@main.command()
def serve():
    """Start the unified ML pipeline server"""
    import subprocess
    import sys

    server_path = Path(__file__).parent.parent / "run_unified_server.py"
    if server_path.exists():
        click.echo("🚀 Starting ML Pipeline Server...")
        subprocess.run([sys.executable, str(server_path)])
    else:
        click.echo("❌ Server file not found!", err=True)


if __name__ == "__main__":
    main()
