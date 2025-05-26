from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="numa_bind",
    sources=["numa_bind.pyx"],
    libraries=["numa"],  # Link with libnuma
    library_dirs=["/usr/lib", "/usr/lib64"],  # Adjust if needed
    include_dirs=["/usr/include"],            # Make sure numa.h is here
)

setup(
    ext_modules=cythonize([ext]),
)
