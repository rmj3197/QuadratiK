from importlib import metadata, import_module

__version__ = "1.1.1"

submodules = [
    "kernel_test",
    "poisson_kernel_test",
    "spherical_clustering",
    "tools",
    "ui",
]
__all__ = submodules + [__version__]


def __dir__():
    return __all__


# taken from scipy
def __getattr__(name):
    if name in submodules:
        return import_module(f"QuadratiK.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'QuadratiK' has no attribute '{name}'")
