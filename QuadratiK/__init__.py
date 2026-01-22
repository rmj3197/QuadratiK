from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("QuadratiK")
except PackageNotFoundError:
    __version__ = "unknown"

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
