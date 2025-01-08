CLASS_REGISTRY = {}


def register_class(cls):
    """
    Register a class to the global registry.
    """
    CLASS_REGISTRY[cls.__name__] = cls
    return cls


def get_registered_class(name):
    """
    Retrieve a registered class by name.
    """
    if name not in CLASS_REGISTRY:
        raise ValueError(f"Class '{name}' is not registered.")
    return CLASS_REGISTRY[name]
