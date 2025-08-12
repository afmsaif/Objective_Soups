#!/usr/bin/env python
"""
Simple script to run training with a configuration
"""
import argparse
import importlib
from train import main

def run_with_config():
    """Run training with a configuration from config.py"""
    parser = argparse.ArgumentParser(description="Run training with configuration")
    parser.add_argument("--config", type=str, default="Config", help="Configuration class name from config.py")
    parser.add_argument("--override", nargs="*", default=[], help="Override config parameters, e.g. num_epochs=20")
    
    args = parser.parse_args()
    
    # Import configuration
    config_module = importlib.import_module("config")
    config_class = getattr(config_module, args.config)
    config_args = config_class.to_args()
    
    # Override parameters if specified
    for override in args.override:
        key, value = override.split("=")
        # Try to infer the type
        try:
            # Try as int
            value = int(value)
        except ValueError:
            try:
                # Try as float
                value = float(value)
            except ValueError:
                # Try as bool
                if value.lower() in ["true", "yes", "1"]:
                    value = True
                elif value.lower() in ["false", "no", "0"]:
                    value = False
                # Otherwise keep as string
        
        # Set the attribute
        setattr(config_args, key, value)
    
    # Run training with the configuration
    main(config_args)

if __name__ == "__main__":
    run_with_config()