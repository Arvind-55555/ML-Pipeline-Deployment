#!/usr/bin/env python3
"""
Simple runner for local ML pipeline development
"""

import click
import yaml
from pathlib import Path
from src.core.orchestrator import MLLocalOrchestrator

@click.group()
def cli():
    """Local ML Pipeline Deployment Tool"""
    pass

@cli.command()
@click.option('--config', default='configs/local_pipeline.yaml', help='Pipeline config file')
@click.option('--data', help='Override data path')
def train(config, data):
    """Train and save a model locally"""
    click.echo("🚀 Starting local ML pipeline...")
    
    orchestrator = MLLocalOrchestrator(config)
    orchestrator.create_pipeline(orchestrator.config)
    
    results = orchestrator.run_pipeline(data)
    
    click.echo(f"✅ Training completed!")
    click.echo(f"📊 Accuracy: {results['training']['metrics']['accuracy']:.3f}")
    click.echo(f"💾 Model: {results['model_path']}")

@cli.command()
@click.option('--model', required=True, help='Path to trained model')
@click.option('--port', default=8000, help='Port for API server')
def serve(model, port):
    """Serve a trained model via local API"""
    click.echo(f"🍽️  Serving model on port {port}...")
    
    orchestrator = MLLocalOrchestrator()
    deployment_id = orchestrator.deploy_model(model, port)
    
    click.echo(f"✅ Model served! Deployment ID: {deployment_id}")
    click.echo(f"🌐 Web UI available at: http://127.0.0.1:{port}")
    click.echo(f"🔗 API available at: http://127.0.0.1:{port}/predict")
    click.echo(f"❤️  Health check: http://127.0.0.1:{port}/health")
    click.echo("Press Ctrl+C to stop the server")

@cli.command()
@click.option('--config', default='configs/local_pipeline.yaml', help='Pipeline config')
def full_deploy(config):
    """Complete pipeline: train + serve"""
    click.echo("🎯 Running complete pipeline: train → serve")
    
    orchestrator = MLLocalOrchestrator(config)
    orchestrator.create_pipeline(orchestrator.config)
    
    # Train
    results = orchestrator.run_pipeline()
    
    # Serve
    deployment_id = orchestrator.deploy_model(results['model_path'])
    
    click.echo(f"🎉 Complete! Model serving at: http://127.0.0.1:8000")
    click.echo(f"🌐 Open your browser to: http://127.0.0.1:8000")
    click.echo(f"   - Web UI: http://127.0.0.1:8000")
    click.echo(f"   - API: http://127.0.0.1:8000/predict")
    click.echo("Press Ctrl+C to stop")

@cli.command()
def status():
    """Check status of running deployments"""
    orchestrator = MLLocalOrchestrator()
    status = orchestrator.get_deployment_status()
    
    if status:
        click.echo("🏃 Running deployments:")
        for dep_id, info in status.items():
            click.echo(f"  {dep_id}: {'running' if info['running'] else 'stopped'}")
    else:
        click.echo("💤 No running deployments")

@cli.command()
def stop():
    """Stop all running deployments"""
    orchestrator = MLLocalOrchestrator()
    orchestrator.stop_servers()
    click.echo("🛑 All deployments stopped")

if __name__ == '__main__':
    cli()