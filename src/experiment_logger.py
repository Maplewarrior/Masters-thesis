import sqlite3
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class ExperimentLogger(ABC):
    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]):
        pass
    
    @abstractmethod
    def finish(self):
        pass

class SQLiteLogger(ExperimentLogger):
    def __init__(self, project_name: str, experiment_name: str):
        self.experiment_name = experiment_name
        self.project_name = project_name
        
        # Setup database
        self.setup_database()
        self.setup_connection()
    
    def setup_database(self):
        conn = sqlite3.connect('experiment_logs.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiment_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            project_name TEXT,
            experiment_name TEXT,
            step INTEGER,
            type TEXT,
            data JSON
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_connection(self):
        self.conn = sqlite3.connect('experiment_logs.db')
        self.cursor = self.conn.cursor()
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.cursor.execute(
            '''INSERT INTO experiment_logs 
               (project_name, experiment_name, step, type, data)
               VALUES (?, ?, ?, ?, ?)''',
            (
                self.project_name,
                self.experiment_name,
                step,
                'metrics',
                json.dumps(metrics)
            )
        )
        self.conn.commit()
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        self.cursor.execute(
            '''INSERT INTO experiment_logs 
               (project_name, experiment_name, type, data)
               VALUES (?, ?, ?, ?)''',
            (
                self.project_name,
                self.experiment_name,
                'hyperparameters',
                json.dumps(params)
            )
        )
        self.conn.commit()
    
    def finish(self):
        self.conn.close()

class WandBLogger(ExperimentLogger):
    def __init__(self, project_name: str, experiment_name: str):
        import wandb
        self.run = wandb.init(
            project=project_name,
            name=experiment_name
        )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.run.log(metrics, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        self.run.config.update(params)
    
    def finish(self):
        self.run.finish()

# Factory to create the appropriate logger
def create_logger(project_name: str, experiment_name: str, backend: str = 'sqlite') -> ExperimentLogger:
    assert backend in ['sqlite', 'wandb'], f"Unknown logger backend: {backend}"
    assert project_name is not None, "Project name is required"
    assert experiment_name is not None, "Experiment name is required"

    if backend.lower() == 'sqlite':
        return SQLiteLogger(project_name, experiment_name)
    elif backend.lower() == 'wandb':
        return WandBLogger(project_name, experiment_name)
    else:
        raise ValueError(f"Unknown logger backend: {backend}")

# Query function for SQLite logs
def query_sqlite_logs(project_name: Optional[str] = None, 
                     experiment_name: Optional[str] = None):
    conn = sqlite3.connect('experiment_logs.db')
    query = 'SELECT * FROM experiment_logs'
    conditions = []
    params = []
    
    if project_name:
        conditions.append('project_name = ?')
        params.append(project_name)
    if experiment_name:
        conditions.append('experiment_name = ?')
        params.append(experiment_name)
    
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Parse JSON data
    df['data'] = df['data'].apply(json.loads)
    return df

# Usage example
if __name__ == '__main__':
    # Configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model_architecture": "resnet50"
    }
    
    # Create logger (easily switch between 'sqlite' and 'wandb')
    logger = create_logger(
        project_name='my_project',
        experiment_name='experiment_001',
        backend='sqlite'  # or 'wandb'
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(config)
    
    # Simulate training loop
    for epoch in range(config['epochs']):
        # Simulate training metrics
        metrics = {
            'train/loss': 0.5 - 0.03 * epoch,
            'train/accuracy': 0.85 + 0.01 * epoch,
            'train/learning_rate': config['learning_rate']
        }
        
        # Log metrics
        logger.log(metrics, step=epoch)
    
    # Cleanup
    logger.finish()

    df = query_sqlite_logs(project_name='my_project', experiment_name='experiment_001')
    print(df)