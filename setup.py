import os
import re
import sys
import platform
import subprocess
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\""
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, 
            cwd=self.build_temp, 
            env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, 
            cwd=self.build_temp
        )


# Read README for long description
with open("python_bindings/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="rodespmm",
    version="0.1.0",
    description="Python bindings for RoDe SpMM (Sparse Matrix-Matrix Multiplication)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    ext_modules=[CMakeExtension("rodespmm")],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(),
) 