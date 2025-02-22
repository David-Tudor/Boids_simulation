from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "boids_omp",
        ["boids_omp.pyx"],
        extra_compile_args=['-fopenmp', '-O3'],
        extra_link_args=['-fopenmp'],
        libraries=["m"]
    )
]

setup(name="boids_omp",
      ext_modules=cythonize(ext_modules))

