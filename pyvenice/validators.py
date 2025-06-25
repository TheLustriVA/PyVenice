"""
Validators for model capability checking and parameter validation.
"""

from functools import wraps
from typing import Dict, Any, Optional, Callable, Set
import warnings

from .exceptions import InvalidRequestError
from .deprecation import check_deprecated_params


def validate_model_capabilities(
    model_param_name: str = "model",
    parameters_to_check: Optional[Set[str]] = None,
    auto_remove_unsupported: bool = True,
    warn_on_removal: bool = True,
):
    """
    Decorator to validate that a model supports the parameters being used.

    Args:
        model_param_name: Name of the model parameter in the function.
        parameters_to_check: Specific parameters to check. If None, checks all known parameters.
        auto_remove_unsupported: Automatically remove unsupported parameters.
        warn_on_removal: Warn when parameters are removed.

    Usage:
        @validate_model_capabilities()
        def create_chat_completion(self, model: str, messages: list, **kwargs):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get the model name from arguments
            model = kwargs.get(model_param_name)
            if not model:
                # Try to get from positional args based on function signature
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if model_param_name in params:
                    idx = params.index(model_param_name)
                    if idx < len(args):
                        model = args[idx]

            if not model:
                # No model specified, can't validate
                return func(self, *args, **kwargs)

            # Get models instance (assumes self has client with models)
            if hasattr(self, "models"):
                models = self.models
            elif hasattr(self, "client") and hasattr(self.client, "models"):
                models = self.client.models
            else:
                # Can't access models, skip validation
                return func(self, *args, **kwargs)

            # Parameters that require specific capabilities
            capability_required_params = {
                "parallel_tool_calls": "supportsFunctionCalling",
                "tools": "supportsFunctionCalling",
                "tool_choice": "supportsFunctionCalling",
                "functions": "supportsFunctionCalling",
                "function_call": "supportsFunctionCalling",
                "response_format": "supportsResponseSchema",
                "response_schema": "supportsResponseSchema",
                "logprobs": "supportsLogProbs",
                "top_logprobs": "supportsLogProbs",
                "reasoning_effort": "supportsReasoning",
            }

            # Check which parameters to validate
            params_to_validate = parameters_to_check or set(
                capability_required_params.keys()
            )

            # Get model capabilities
            capabilities = models.get_capabilities(model)
            if not capabilities:
                # Can't get capabilities, proceed without validation
                return func(self, *args, **kwargs)

            # Check for deprecated parameters first
            # Infer schema name from function/class context
            schema_name = getattr(self, "_schema_name", "ChatCompletionRequest")
            if hasattr(self, "__class__") and "Chat" in self.__class__.__name__:
                schema_name = "ChatCompletionRequest"
            elif hasattr(self, "__class__") and "Image" in self.__class__.__name__:
                schema_name = "GenerateImageRequest"

            # Filter deprecated parameters
            kwargs = check_deprecated_params(schema_name, **kwargs)

            # Check each parameter for capability support
            unsupported_params = []
            for param, capability_field in capability_required_params.items():
                if param not in params_to_validate:
                    continue

                if param in kwargs and kwargs[param] is not None:
                    if not getattr(capabilities, capability_field, False):
                        unsupported_params.append(param)

            if unsupported_params:
                if auto_remove_unsupported:
                    # Remove unsupported parameters
                    for param in unsupported_params:
                        if warn_on_removal:
                            warnings.warn(
                                f"Parameter '{param}' is not supported by model '{model}' and has been removed from the request.",
                                UserWarning,
                                stacklevel=2,
                            )
                        kwargs.pop(param, None)
                else:
                    # Raise error
                    raise InvalidRequestError(
                        f"Model '{model}' does not support the following parameters: {', '.join(unsupported_params)}. "
                        f"Check model capabilities before using these parameters."
                    )

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def filter_unsupported_params(
    model: str,
    params: Dict[str, Any],
    models_instance: Any,
    warn: bool = True,
) -> Dict[str, Any]:
    """
    Filter out parameters that are not supported by a model.

    Args:
        model: The model ID.
        params: Parameters to filter.
        models_instance: Instance of Models class to check capabilities.
        warn: Whether to warn about removed parameters.

    Returns:
        Filtered parameters dictionary.
    """
    capabilities = models_instance.get_capabilities(model)
    if not capabilities:
        # Can't validate, return params as-is
        return params

    capability_required_params = {
        "parallel_tool_calls": "supportsFunctionCalling",
        "tools": "supportsFunctionCalling",
        "tool_choice": "supportsFunctionCalling",
        "functions": "supportsFunctionCalling",
        "function_call": "supportsFunctionCalling",
        "response_format": "supportsResponseSchema",
        "response_schema": "supportsResponseSchema",
        "logprobs": "supportsLogProbs",
        "top_logprobs": "supportsLogProbs",
        "reasoning_effort": "supportsReasoning",
    }

    filtered = params.copy()

    for param, capability_field in capability_required_params.items():
        if param in filtered and filtered[param] is not None:
            if not getattr(capabilities, capability_field, False):
                if warn:
                    warnings.warn(
                        f"Parameter '{param}' is not supported by model '{model}' and has been removed.",
                        UserWarning,
                        stacklevel=2,
                    )
                filtered.pop(param)

    return filtered
