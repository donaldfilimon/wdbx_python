"""
Import utilities for WDBX to handle circular imports and dynamic module loading.

This module provides tools to safely handle circular imports by using lazy imports
and runtime dependency resolution.
"""

import importlib
import inspect
import os
import sys
import time
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

from .logging_utils import get_logger

logger = get_logger("WDBX.ImportUtils")

# Type variables for generic function signatures
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Registry of lazy imports
_LAZY_IMPORTS: Dict[str, Dict[str, Any]] = {}
_IMPORT_TIMES: Dict[str, float] = {}
_IMPORT_DEPENDENCIES: Dict[str, Set[str]] = {}
_CIRCULAR_IMPORTS_DETECTED: List[Tuple[str, str]] = []
_IMPORT_STACK: List[str] = []


def lazy_import(module_name: str, names: Optional[List[str]] = None) -> Union[Any, Dict[str, Any]]:
    """
    Lazily import a module or specific objects from a module.
    
    Args:
        module_name: The name of the module to import
        names: Optional list of specific names to import from the module
        
    Returns:
        If names is None, returns a proxy module. If names is provided,
        returns a dictionary mapping names to objects from the module.
    """
    # Check if we've already imported this module
    if module_name in _LAZY_IMPORTS and names is None:
        return _LAZY_IMPORTS[module_name]
    
    # Track import dependencies
    current_module = _get_caller_module()
    if current_module:
        if current_module not in _IMPORT_DEPENDENCIES:
            _IMPORT_DEPENDENCIES[current_module] = set()
        _IMPORT_DEPENDENCIES[current_module].add(module_name)
        
        # Check for circular imports
        if module_name in _IMPORT_STACK:
            circular_path = ' -> '.join(_IMPORT_STACK[_IMPORT_STACK.index(module_name):] + [module_name])
            circular_import = (current_module, module_name)
            if circular_import not in _CIRCULAR_IMPORTS_DETECTED:
                _CIRCULAR_IMPORTS_DETECTED.append(circular_import)
                logger.warning(f"Circular import detected: {circular_path}")
    
    if names is None:
        # Create a proxy module that will be populated on first access
        proxy = _LazyModule(module_name)
        _LAZY_IMPORTS[module_name] = proxy
        return proxy
    else:
        # Import specific names from the module
        result = {}
        for name in names:
            result[name] = _LazyObject(module_name, name)
        return result


def lazy_class_attribute(attr_name: str, class_to_import: str, module_to_import: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to lazily import a class attribute to avoid circular imports.
    
    Args:
        attr_name: The name of the attribute to lazily import
        class_to_import: The name of the class to import
        module_to_import: The name of the module to import from
        
    Returns:
        A decorator function
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Store the original __getattr__ if it exists
        original_getattr = cls.__getattr__ if hasattr(cls, '__getattr__') else None
        
        def __getattr__(self, name):
            if name == attr_name:
                # Dynamically import the class when the attribute is accessed
                module = importlib.import_module(module_to_import)
                imported_class = getattr(module, class_to_import)
                # Cache the imported class on the instance
                setattr(self, attr_name, imported_class)
                return imported_class
            elif original_getattr:
                return original_getattr(self, name)
            else:
                raise AttributeError(f"{cls.__name__} has no attribute '{name}'")
        
        # Add the __getattr__ method to the class
        cls.__getattr__ = __getattr__
        return cls
    
    return decorator


def lazy_property(import_path: str) -> Callable[[Any], property]:
    """
    Create a property that lazily imports a value.
    
    Args:
        import_path: Module and attribute path, e.g. 'wdbx.ml.backend.MLBackend'
        
    Returns:
        A property that lazily imports the value
    """
    module_path, attr_name = import_path.rsplit('.', 1)
    
    def getter(self):
        # Import the module and get the attribute when the property is accessed
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        
        # Cache the value on the instance using a private attribute
        private_name = f"_{attr_name}"
        setattr(self, private_name, value)
        
        return value
    
    return property(getter)


def runtime_import(import_path: str) -> Any:
    """
    Import a module or object at runtime.
    
    Args:
        import_path: Module path or module.attribute path
        
    Returns:
        Imported module or object
    """
    start_time = time.time()
    
    try:
        if '.' in import_path:
            module_path, attr_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            
            # Add to import stack for circular detection
            _IMPORT_STACK.append(module_path)
            try:
                result = getattr(module, attr_name)
            finally:
                if _IMPORT_STACK and _IMPORT_STACK[-1] == module_path:
                    _IMPORT_STACK.pop()
        else:
            # Add to import stack for circular detection
            _IMPORT_STACK.append(import_path)
            try:
                result = importlib.import_module(import_path)
            finally:
                if _IMPORT_STACK and _IMPORT_STACK[-1] == import_path:
                    _IMPORT_STACK.pop()
        
        # Record import time
        elapsed = time.time() - start_time
        _IMPORT_TIMES[import_path] = elapsed
        
        if elapsed > 0.1:  # Log slow imports (>100ms)
            logger.debug(f"Slow import: {import_path} took {elapsed:.3f}s")
        
        return result
    except ImportError as e:
        logger.error(f"Failed to import {import_path}: {e}")
        raise


def import_later(func: F) -> F:
    """
    Decorator to delay imports until function is called.
    
    Parses import statements from function docstring and executes them at runtime.
    
    Example:
        @import_later
        def my_function():
            '''
            IMPORTS:
            from module1 import Class1
            import module2
            '''
            # Function code using Class1 and module2
            
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get docstring
        doc = func.__doc__
        if doc and 'IMPORTS:' in doc:
            # Extract import statements
            import_section = doc.split('IMPORTS:')[1].split('\n')
            import_statements = [line.strip() for line in import_section 
                                 if line.strip() and not line.strip().startswith('#')]
            
            # Get function's globals
            func_globals = func.__globals__
            
            # Execute imports in the function's global namespace
            for statement in import_statements:
                try:
                    if statement.startswith('from '):
                        # Parse "from X import Y, Z"
                        module_part, names_part = statement.replace('from ', '').split(' import ')
                        module = runtime_import(module_part)
                        for name in [n.strip() for n in names_part.split(',')]:
                            if ' as ' in name:
                                orig_name, as_name = [n.strip() for n in name.split(' as ')]
                                func_globals[as_name] = getattr(module, orig_name)
                            else:
                                func_globals[name] = getattr(module, name)
                    elif statement.startswith('import '):
                        # Parse "import X as Y" or "import X, Y"
                        for module_spec in statement.replace('import ', '').split(','):
                            module_spec = module_spec.strip()
                            if ' as ' in module_spec:
                                module_name, as_name = [n.strip() for n in module_spec.split(' as ')]
                                func_globals[as_name] = runtime_import(module_name)
                            else:
                                func_globals[module_spec] = runtime_import(module_spec)
                except Exception as e:
                    logger.error(f"Error in deferred import for {func.__name__}: {statement} - {e}")
                    raise
        
        # Call the original function
        return func(*args, **kwargs)
    
    return cast(F, wrapper)


def get_import_times() -> Dict[str, float]:
    """
    Get a dictionary of module import times.
    
    Returns:
        Dictionary mapping module names to import times in seconds
    """
    return dict(_IMPORT_TIMES)


def get_import_dependencies() -> Dict[str, List[str]]:
    """
    Get a dictionary of module dependencies.
    
    Returns:
        Dictionary mapping module names to lists of imported modules
    """
    return {k: list(v) for k, v in _IMPORT_DEPENDENCIES.items()}


def get_circular_imports() -> List[Tuple[str, str]]:
    """
    Get a list of detected circular imports.
    
    Returns:
        List of (importing_module, imported_module) tuples
    """
    return list(_CIRCULAR_IMPORTS_DETECTED)


def visualize_dependencies() -> Dict[str, Any]:
    """
    Create a visualization of module dependencies.
    
    Returns:
        Dictionary with nodes and edges for visualization
    """
    nodes = []
    edges = []
    
    # Create nodes for all modules
    modules = set()
    for module in _IMPORT_DEPENDENCIES:
        modules.add(module)
        for dep in _IMPORT_DEPENDENCIES[module]:
            modules.add(dep)
    
    for module in sorted(modules):
        # Calculate node size based on number of dependents
        dependents = sum(1 for m, deps in _IMPORT_DEPENDENCIES.items() if module in deps)
        nodes.append({
            "id": module,
            "label": module.split('.')[-1],
            "full_name": module,
            "size": 10 + dependents * 2,
            "color": "#ff0000" if any(m == module for m, d in _CIRCULAR_IMPORTS_DETECTED) else "#1f77b4"
        })
    
    # Create edges for dependencies
    for module, deps in _IMPORT_DEPENDENCIES.items():
        for dep in deps:
            is_circular = any((module == m and dep == d) or (module == d and dep == m) 
                             for m, d in _CIRCULAR_IMPORTS_DETECTED)
            edges.append({
                "source": module,
                "target": dep,
                "color": "#ff0000" if is_circular else "#cccccc"
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "circular_imports": _CIRCULAR_IMPORTS_DETECTED
    }


# Private helpers

class _LazyModule:
    """Proxy for a module that is imported on first attribute access."""
    
    def __init__(self, name: str):
        self.__dict__['_module_name'] = name
        self.__dict__['_module'] = None
    
    def __getattr__(self, name: str) -> Any:
        if self.__dict__['_module'] is None:
            # Import the module when an attribute is first accessed
            _IMPORT_STACK.append(self.__dict__['_module_name'])
            try:
                self.__dict__['_module'] = runtime_import(self.__dict__['_module_name'])
            finally:
                if _IMPORT_STACK and _IMPORT_STACK[-1] == self.__dict__['_module_name']:
                    _IMPORT_STACK.pop()
        
        return getattr(self.__dict__['_module'], name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__['_module'] is None:
            # Import the module when an attribute is first set
            _IMPORT_STACK.append(self.__dict__['_module_name'])
            try:
                self.__dict__['_module'] = runtime_import(self.__dict__['_module_name'])
            finally:
                if _IMPORT_STACK and _IMPORT_STACK[-1] == self.__dict__['_module_name']:
                    _IMPORT_STACK.pop()
        
        setattr(self.__dict__['_module'], name, value)


class _LazyObject:
    """Proxy for an object from a module that is imported on first access."""
    
    def __init__(self, module_name: str, object_name: str):
        self.__dict__['_module_name'] = module_name
        self.__dict__['_object_name'] = object_name
        self.__dict__['_object'] = None
    
    def __call__(self, *args, **kwargs):
        if self.__dict__['_object'] is None:
            # Import the object when it is first called
            _IMPORT_STACK.append(self.__dict__['_module_name'])
            try:
                module = runtime_import(self.__dict__['_module_name'])
                self.__dict__['_object'] = getattr(module, self.__dict__['_object_name'])
            finally:
                if _IMPORT_STACK and _IMPORT_STACK[-1] == self.__dict__['_module_name']:
                    _IMPORT_STACK.pop()
        
        return self.__dict__['_object'](*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        if self.__dict__['_object'] is None:
            # Import the object when an attribute is first accessed
            _IMPORT_STACK.append(self.__dict__['_module_name'])
            try:
                module = runtime_import(self.__dict__['_module_name'])
                self.__dict__['_object'] = getattr(module, self.__dict__['_object_name'])
            finally:
                if _IMPORT_STACK and _IMPORT_STACK[-1] == self.__dict__['_module_name']:
                    _IMPORT_STACK.pop()
        
        return getattr(self.__dict__['_object'], name)


def _get_caller_module() -> Optional[str]:
    """Get the name of the calling module."""
    frame = inspect.currentframe()
    if frame is None:
        return None
    
    # Get the caller's frame (2 frames up from current frame)
    caller_frame = frame.f_back
    if caller_frame is None:
        return None
    caller_frame = caller_frame.f_back
    if caller_frame is None:
        return None
    
    # Get the module name from the frame's globals
    module_globals = caller_frame.f_globals
    return module_globals.get('__name__') 