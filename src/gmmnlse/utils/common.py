"""Common utility functions for GMMNLSE solver.

This module provides shared helper functions used across multiple modules
to reduce code duplication and ensure consistency.
"""

from typing import Union
import jax.numpy as jnp
from jax import core as jax_core

# Type aliases for clarity
FloatLike = Union[float, jnp.ndarray]
ScalarLike = Union[float, jnp.ndarray]


def _return_float_if_possible(value: FloatLike) -> FloatLike:
    """Return Python float unless the value is a JAX tracer.
    
    This function is essential for methods that need to return Python floats
    when called outside JAX transformations (e.g., for plotting or user output),
    but must preserve JAX tracers when called inside JAX transformations
    (e.g., jax.jit, jax.grad) to maintain computational graphs.
    
    Args:
        value: A scalar value that may be a Python float or a JAX tracer.
    
    Returns:
        If value is a JAX tracer, returns it unchanged. Otherwise, converts
        to Python float.
    
    Example:
        >>> result = _return_float_if_possible(jnp.array(5.0))
        >>> type(result)
        <class 'float'>
        
        >>> # Inside jax.jit, tracers are preserved
        >>> @jax.jit
        >>> def f(x):
        ...     return _return_float_if_possible(x * 2)
    """
    if isinstance(value, jax_core.Tracer):
        return value
    return float(value)

