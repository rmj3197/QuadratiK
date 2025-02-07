from importlib import import_module

__version__ = "1.1.3dev0"

submodules = [
    "kernel_test",
    "poisson_kernel_test",
    "spherical_clustering",
    "tools",
    "ui",
]
__all__ = submodules


def __dir__():
    return __all__


# taken from scipy
def __getattr__(name):
    if name in submodules:
        return import_module(f"QuadratiK.{name}")
    else:
        try:
            return globals()[name]
        except KeyError as err:
            raise AttributeError(
                f"Module 'QuadratiK' has no attribute '{name}'"
            ) from err
