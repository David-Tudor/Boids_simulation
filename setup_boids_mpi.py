from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "boids_mpi",
        ["boids_mpi.pyx"],
        libraries=["m", "mpi"]
    )
]

setup(name="boids_mpi",
      ext_modules=cythonize(ext_modules))

