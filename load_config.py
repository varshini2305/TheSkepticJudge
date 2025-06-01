import yaml
import os

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_env_vars():
    """Get environment variables from config."""
    config = load_config()
    env_vars = config.get('env', {})
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars
