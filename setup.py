import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import site
import sysconfig
import shutil

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Find nanobind
        nanobind_path = None
        for path in site.getsitepackages():
            candidate = os.path.join(path, 'nanobind', 'cmake')
            if os.path.exists(os.path.join(candidate, 'nanobind-config.cmake')):
                nanobind_path = candidate
                break

        if nanobind_path is None:
            raise RuntimeError("Could not find nanobind installation.")

        # Get Python information
        python_include = sysconfig.get_path('include')
        python_lib = sysconfig.get_config_var('LIBDIR')

        # Create build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_INCLUDE_DIR={python_include}",
            f"-DPYTHON_LIBRARY={python_lib}",
            f"-Dnanobind_DIR={nanobind_path}",
        ]

        # Find Ninja
        ninja_path = shutil.which("ninja")
        if ninja_path:
            print(f"Found Ninja at: {ninja_path}")
            cmake_args.extend(["-GNinja", f"-DCMAKE_MAKE_PROGRAM={ninja_path}"])
        else:
            print("Ninja not found, using default CMake generator")

        build_args = [
            "--config", "Release",
            "--target", "jobshop"
        ]

        env = os.environ.copy()
        env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\""

        preset_file = os.path.join(ext.sourcedir, "jsp", "CMakePresets.json")
        if os.path.exists(preset_file):
            cmake_args.extend(["--preset", "pip-install"])

        # Configure
        subprocess.check_call(["cmake", os.path.join(ext.sourcedir, "jsp")] + cmake_args, cwd=build_temp, env=env)


        # Build
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

setup(
    name="jobshop",
    version="0.1.0",
    author="Per-Arne Andersen",
    author_email="per@sysx.no",
    description="Job Shop Scheduling Algorithms",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("jobshop")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10",
)