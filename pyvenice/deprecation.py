"""
Deprecation Warning System for Venice.ai API Changes

Handles deprecated parameters and provides migration guidance.
"""

import warnings
from functools import wraps
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path


class DeprecationManager:
    """Manages deprecated parameters and API features"""
    
    def __init__(self):
        self.deprecated_params = self._load_deprecated_params()
    
    def _load_deprecated_params(self) -> Dict[str, Any]:
        """Load deprecated parameters from configuration"""
        config_path = Path(__file__).parent.parent / "docs" / "deprecated_params.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def is_deprecated(self, schema: str, parameter: str) -> bool:
        """Check if a parameter is deprecated"""
        return schema in self.deprecated_params and parameter in self.deprecated_params[schema]
    
    def get_deprecation_info(self, schema: str, parameter: str) -> Optional[Dict[str, Any]]:
        """Get deprecation information for a parameter"""
        if self.is_deprecated(schema, parameter):
            return self.deprecated_params[schema][parameter]
        return None
    
    def warn_if_deprecated(self, schema: str, **kwargs):
        """Issue warnings for any deprecated parameters in kwargs"""
        for param_name, param_value in kwargs.items():
            if self.is_deprecated(schema, param_name):
                info = self.get_deprecation_info(schema, param_name)
                message = f"Parameter '{param_name}' is deprecated"
                
                if info.get('removed_version'):
                    message += f" (removed in API version {info['removed_version']})"
                
                if info.get('replacement'):
                    message += f". Use '{info['replacement']}' instead"
                
                if info.get('removal_date'):
                    message += f". Support will be removed after {info['removal_date']}"
                
                warnings.warn(
                    message,
                    DeprecationWarning,
                    stacklevel=3
                )
    
    def filter_deprecated_params(self, schema: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out deprecated parameters from a params dict"""
        filtered = {}
        for key, value in params.items():
            if self.is_deprecated(schema, key):
                info = self.get_deprecation_info(schema, key)
                if info.get('behavior') == 'remove':
                    # Parameter was removed from API, don't send it
                    continue
                elif info.get('behavior') == 'warn':
                    # Parameter still works but is deprecated
                    self.warn_if_deprecated(schema, **{key: value})
            
            filtered[key] = value
        
        return filtered


# Global instance
deprecation_manager = DeprecationManager()


def deprecated_parameter(schema: str, parameter: str, removed_version: str = None, 
                        replacement: str = None, removal_date: str = None,
                        behavior: str = "warn"):
    """
    Decorator to mark a parameter as deprecated
    
    Args:
        schema: The schema name (e.g., 'ChatCompletionRequest')
        parameter: The parameter name
        removed_version: API version where parameter was removed
        replacement: Suggested replacement parameter
        removal_date: When support will be removed from client
        behavior: 'warn' (send with warning) or 'remove' (filter out)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check for deprecated parameter
            if parameter in kwargs:
                message = f"Parameter '{parameter}' is deprecated"
                if removed_version:
                    message += f" (removed in API version {removed_version})"
                if replacement:
                    message += f". Use '{replacement}' instead"
                if removal_date:
                    message += f". Support will be removed after {removal_date}"
                
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                
                # Remove parameter if it was removed from API
                if behavior == 'remove':
                    kwargs.pop(parameter)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_deprecated_params(schema: str, **kwargs) -> Dict[str, Any]:
    """
    Utility function to check for deprecated parameters and filter them
    
    Args:
        schema: The schema name to check against
        **kwargs: Parameters to check
    
    Returns:
        Filtered parameters dict with deprecated params handled
    """
    return deprecation_manager.filter_deprecated_params(schema, kwargs)