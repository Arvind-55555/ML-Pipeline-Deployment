# run_real_time.py

import click
import asyncio
from src.core.orchestrator import MLLocalOrchestrator
from examples.real_time_stocks.stock_pipeline import StockPredictionPipeline
from examples.iot_sensors.sensor_pipeline import IoTSensorPipeline

@click.group()
def cli():
    """Real-time ML Pipeline Deployment"""
    pass

@cli.command()
@click.option('--symbols', multiple=True, help='Stock symbols to monitor (use multiple times: --symbols AAPL --symbols GOOGL, or comma-separated: --symbols AAPL,GOOGL)')
@click.option('--interval', default=60, help='Update interval in seconds')
def stocks(symbols, interval):
    """Start real-time stock prediction pipeline"""
    click.echo("📈 Starting real-time stock prediction...")
    
    pipeline = StockPredictionPipeline()
    if symbols:
        # Convert tuple to list and handle comma-separated values
        symbol_list = []
        for sym in symbols:
            # Allow comma-separated values in a single argument
            symbol_list.extend([s.strip().upper() for s in str(sym).split(',') if s.strip()])
        pipeline.config['symbols'] = symbol_list if symbol_list else [s.upper() for s in symbols]
    
    asyncio.run(pipeline.start_real_time_pipeline())

@cli.command()
@click.option('--sensors', default=10, help='Number of sensors to simulate')
def iot(sensors):
    """Start IoT sensor analytics pipeline"""
    click.echo("🏭 Starting IoT sensor analytics...")
    
    pipeline = IoTSensorPipeline()
    asyncio.run(pipeline.simulate_sensor_data(sensors))

@cli.command()
@click.option('--config', required=True, help='Real-time pipeline config')
def start_realtime(config):
    """Start real-time ML pipeline with configuration"""
    click.echo("🚀 Starting real-time ML pipeline...")
    
    orchestrator = MLLocalOrchestrator(config)
    orchestrator.create_pipeline(orchestrator.config)
    
    # Create continuous training pipeline
    from src.stages.continuous_training import ContinuousTrainingPipeline
    
    def fetch_latest_data():
        """Fetch latest data for continuous training"""
        # This should be implemented based on your data source
        # For now, return empty data as placeholder
        import pandas as pd
        import numpy as np
        # Placeholder - replace with actual data fetching
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        return X, y
    
    def load_validation_data():
        """Load validation data"""
        # This should load actual validation data
        # For now, return placeholder
        import pandas as pd
        import numpy as np
        X_val = np.random.rand(50, 4)
        y_val = np.random.randint(0, 3, 50)
        return X_val, y_val
    
    continuous_training = ContinuousTrainingPipeline(orchestrator.config)
    continuous_training.start_continuous_training(
        data_fetcher=fetch_latest_data,
        validation_data=load_validation_data()
    )
    
    click.echo("✅ Real-time pipeline started")
    click.echo("Press Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        continuous_training.stop_continuous_training()
        click.echo("\n🛑 Pipeline stopped")

@cli.command()
def monitor():
    """Monitor real-time pipeline performance"""
    click.echo("📊 Real-time Pipeline Monitoring")
    click.echo("=" * 40)
    
    # Display current metrics
    # - Throughput (predictions/second)
    # - Latency (ms)
    # - Model accuracy
    # - Data drift metrics
    # - System resources

if __name__ == '__main__':
    cli()