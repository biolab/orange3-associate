
from distutils.core import setup, Extension

setup(
    ext_modules=[
        Extension("_fpgrowth",
                  sources=["_fpgrowth.cpp"],
                  extra_compile_args=["-std=c++11", "-O3"],
                  language="c++",)
    ],
)
