"""
Configuration loaders for YAML files with environment variable substitution
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import re

class ConfigLoader:
    """Loads and parses YAML configuration files with env variable substitution"""
    
    @staticmethod
    def load_yaml_config(file_path: Path, env_substitute: bool = True) -> Dict[str, Any]:
        """
        Load YAML configuration file with optional environment variable substitution
        
        Args:
            file_path: Path to YAML file
            env_substitute: Whether to substitute environment variables
            
        Returns:
            Dictionary with configuration
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if env_substitute:
            content = ConfigLoader._substitute_env_vars(content)
            
        config = yaml.safe_load(content)
        return config or {}
    
    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        def replace_env_var(match):
            var_name = match.group(1)
            # Handle default values: ${VAR:-default}
            if ':-' in var_name:
                var_name, default = var_name.split(':-', 1)
                value = os.getenv(var_name, default)
            else:
                value = os.getenv(var_name, '')
            return str(value)
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([A-Za-z0-9_:-]+)\}'
        return re.sub(pattern, replace_env_var, content)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Override configuration (takes precedence)
            
        Returns:
            Merged configuration
        """
        def _deep_merge(base: Any, override: Any) -> Any:
            if isinstance(base, dict) and isinstance(override, dict):
                result = base.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = _deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            else:
                return override
        
        return _deep_merge(base_config, override_config)
    
    @staticmethod
    def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Args:
            config: Configuration dictionary
            path: Dot notation path (e.g., "audio.input.device")
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


# Convenience functions
def load_yaml_config(file_path: Path, env_substitute: bool = True) -> Dict[str, Any]:
    return ConfigLoader.load_yaml_config(file_path, env_substitute)

def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    return ConfigLoader.merge_configs(base_config, override_config)

def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    return ConfigLoader.get_nested_config(config, path, default)