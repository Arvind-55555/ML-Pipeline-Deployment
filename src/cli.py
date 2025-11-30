import click
from core.orchestrator import MLPipelineOrchestrator

@click.group()
def cli():
    """ML Pipeline Deployment Tool"""
    pass

@cli.command()
@click.option('--config', required=True, help='Pipeline config file')
@click.option('--type', default='sklearn', help='Pipeline type')
def deploy(config, type):
    """Deploy ML pipeline"""
    orchestrator = MLPipelineOrchestrator(config)
    deployment_id = orchestrator.deploy_pipeline(type)
    click.echo(f"Deployed: {deployment_id}")

@cli.command()
@click.option('--deployment-id', required=True, help='Deployment ID')
@click.option('--replicas', required=True, type=int, help='Number of replicas')
def scale(deployment_id, replicas):
    """Scale pipeline deployment"""
    orchestrator = MLPipelineOrchestrator()
    orchestrator.scale_pipeline(deployment_id, replicas)
    click.echo(f"Scaled to {replicas} replicas")

if __name__ == '__main__':
    cli()