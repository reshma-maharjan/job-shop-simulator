import os
import sys
from setuptools import setup

# Check if we want Python-only installation
PYTHON_ONLY = os.environ.get("PYTHON_ONLY", "0").lower() in ("1", "true", "yes")

if not PYTHON_ONLY:
    import subprocess
    import site
    import shutil
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext
    import tomli

    def load_config():
        """Load configuration from pyproject.toml."""
        try:
            with open("pyproject.toml", "rb") as f:
                pyproject = tomli.load(f)
                return {
                    'vcpkg': pyproject.get("tool", {}).get("vcpkg", {}),
                    'cmake': pyproject.get("tool", {}).get("cmake", {})
                }
        except FileNotFoundError:
            print("Warning: pyproject.toml not found")
            return {'vcpkg': {}, 'cmake': {}}

    class CMakeExtension(Extension):
        def __init__(self, name, sourcedir=""):
            Extension.__init__(self, name, sources=[])
            self.sourcedir = os.path.abspath(os.path.dirname(__file__))

    class CMakeBuild(build_ext):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            config = load_config()
            self.vcpkg_config = config['vcpkg']
            self.cmake_config = config['cmake']

        def run(self):
            try:
                subprocess.check_output(["cmake", "--version"])
            except OSError:
                raise RuntimeError(
                    "CMake must be installed to build the following extensions: "
                    + ", ".join(e.name for e in self.extensions)
                )

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            self._setup_vcpkg()

            for ext in self.extensions:
                self.build_extension(ext)

        def _setup_vcpkg(self):
            """Setup vcpkg with configuration from pyproject.toml."""
            vcpkg_root = os.path.abspath(os.path.join(self.build_temp, '.vcpkg'))
            os.environ['VCPKG_ROOT'] = vcpkg_root

            repository = self.vcpkg_config.get("repository", "https://github.com/Microsoft/vcpkg.git")
            revision = self.vcpkg_config.get("revision")
            manifest_path = self.vcpkg_config.get("manifest")

            try:
                if os.path.exists(vcpkg_root):
                    shutil.rmtree(vcpkg_root)

                print(f"Cloning vcpkg repository...")
                subprocess.check_call(
                    ["git", "clone", repository, vcpkg_root],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if revision:
                    print(f"Checking out revision {revision}")
                    subprocess.check_call(
                        ["git", "fetch", "--all"],
                        cwd=vcpkg_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    subprocess.check_call(
                        ["git", "reset", "--hard", revision],
                        cwd=vcpkg_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                if manifest_path:
                    manifest_path = os.path.abspath(manifest_path)
                    if os.path.exists(manifest_path):
                        print(f"Using vcpkg.json from {manifest_path}")
                        vcpkg_json_target = os.path.join(vcpkg_root, 'vcpkg.json')
                        shutil.copy2(manifest_path, vcpkg_json_target)
                    else:
                        print(f"Warning: Specified vcpkg.json at {manifest_path} not found")

                print("Bootstrapping vcpkg...")
                bootstrap_script = 'bootstrap-vcpkg.bat' if sys.platform.startswith('win32') else 'bootstrap-vcpkg.sh'
                bootstrap_path = os.path.join(vcpkg_root, bootstrap_script)

                if not os.path.exists(bootstrap_path):
                    raise RuntimeError(f"Bootstrap script not found at {bootstrap_path}")

                if not sys.platform.startswith('win32'):
                    os.chmod(bootstrap_path, 0o755)

                result = subprocess.run(
                    [os.path.abspath(bootstrap_path)],
                    cwd=vcpkg_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    print("Bootstrap stdout:", result.stdout)
                    print("Bootstrap stderr:", result.stderr)
                    raise RuntimeError("vcpkg bootstrap failed")
                else:
                    print("vcpkg bootstrap successful")

                if manifest_path and os.path.exists(os.path.join(vcpkg_root, 'vcpkg.json')):
                    self._install_manifest_mode(vcpkg_root)
                else:
                    self._install_vcpkg_packages(vcpkg_root)

            except subprocess.CalledProcessError as e:
                print(f"Command failed: {e.cmd}")
                print(f"Output: {e.output.decode() if e.output else 'No output'}")
                print(f"Error: {e.stderr.decode() if e.stderr else 'No error output'}")
                raise RuntimeError("Failed to setup vcpkg") from e
            except Exception as e:
                print(f"Error setting up vcpkg: {str(e)}")
                raise RuntimeError("Failed to setup vcpkg") from e

        def _install_manifest_mode(self, vcpkg_root):
            """Install packages using vcpkg manifest mode."""
            vcpkg_exe = os.path.abspath(os.path.join(
                vcpkg_root,
                'vcpkg.exe' if sys.platform.startswith('win32') else './vcpkg'
            ))

            if not os.path.exists(vcpkg_exe) and not sys.platform.startswith('win32'):
                vcpkg_exe = os.path.abspath(os.path.join(vcpkg_root, 'vcpkg'))
                if not os.path.exists(vcpkg_exe):
                    raise RuntimeError(f"vcpkg executable not found at {vcpkg_exe}")

            if not sys.platform.startswith('win32'):
                os.chmod(vcpkg_exe, 0o755)

            print("Installing dependencies from vcpkg.json...")
            result = subprocess.run(
                [vcpkg_exe, "install", "--clean-after-build"],
                cwd=vcpkg_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print("Installation failed")
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
                raise RuntimeError("Failed to install vcpkg dependencies")
            else:
                print("Successfully installed all dependencies from vcpkg.json")

            self._integrate_vcpkg(vcpkg_exe, vcpkg_root)

        def _install_vcpkg_packages(self, vcpkg_root):
            """Install vcpkg packages based on platform."""
            vcpkg_exe = os.path.join(vcpkg_root, 'vcpkg.exe' if sys.platform.startswith('win32') else 'vcpkg')
            vcpkg_exe = os.path.abspath(vcpkg_exe)

            if not os.path.exists(vcpkg_exe) and not sys.platform.startswith('win32'):
                raise RuntimeError(f"vcpkg executable not found at {vcpkg_exe}")

            if not sys.platform.startswith('win32'):
                os.chmod(vcpkg_exe, 0o755)

            packages = self.vcpkg_config.get("packages", {}).get("common", [])
            if sys.platform.startswith('win32'):
                packages.extend(self.vcpkg_config.get("packages", {}).get("windows", []))
            elif sys.platform.startswith('darwin'):
                packages.extend(self.vcpkg_config.get("packages", {}).get("macos", []))
            else:
                packages.extend(self.vcpkg_config.get("packages", {}).get("linux", []))

            for package in packages:
                try:
                    print(f"Installing {package}...")
                    result = subprocess.run(
                        [vcpkg_exe, "install", package, "--clean-after-build"],
                        cwd=vcpkg_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.returncode != 0:
                        print(f"Warning: Failed to install {package}")
                        print("stdout:", result.stdout)
                        print("stderr:", result.stderr)
                        raise RuntimeError(f"Failed to install {package}")
                    else:
                        print(f"Successfully installed {package}")
                except Exception as e:
                    print(f"Error installing {package}: {str(e)}")
                    raise RuntimeError(f"Failed to install vcpkg package {package}") from e

            self._integrate_vcpkg(vcpkg_exe, vcpkg_root)

        def _integrate_vcpkg(self, vcpkg_exe, vcpkg_root):
            """Integrate vcpkg with the system."""
            try:
                result = subprocess.run(
                    [vcpkg_exe, "integrate", "install"],
                    cwd=vcpkg_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    print("Warning: vcpkg integration failed")
                    print("stdout:", result.stdout)
                    print("stderr:", result.stderr)
                else:
                    print("vcpkg integration successful")
            except Exception as e:
                print(f"Warning: vcpkg integration failed: {str(e)}")

        def build_extension(self, ext):
            """Build the extension using CMake."""
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            build_temp = os.path.join(self.build_temp, ext.name)

            cmake_args, source_dir = self._get_cmake_args(ext, extdir, self._find_nanobind())
            build_args = ["--config", self.cmake_config.get('build_type', 'Release')]

            if target := self.cmake_config.get('target'):
                build_args.extend(["--target", target])

            os.makedirs(build_temp, exist_ok=True)

            self._configure_cmake(ext, cmake_args, build_temp, self._get_build_env())
            self._build_cmake(build_args, build_temp)

        def _find_nanobind(self):
            """Find nanobind installation path."""
            for path in site.getsitepackages():
                candidate = os.path.join(path, 'nanobind', 'cmake')
                if os.path.exists(os.path.join(candidate, 'nanobind-config.cmake')):
                    return candidate
            raise RuntimeError("Could not find nanobind installation.")

        def _get_cmake_args(self, ext, extdir, nanobind_path):
            source_dir = os.path.abspath(os.path.join(
                ext.sourcedir,
                self.cmake_config.get("source_dir", "per_jsp/cpp")
            ))

            args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dnanobind_DIR={nanobind_path}",
                f"-DCMAKE_TOOLCHAIN_FILE={os.path.join(os.environ['VCPKG_ROOT'], 'scripts/buildsystems/vcpkg.cmake')}",
                f"-DCMAKE_BUILD_TYPE={self.cmake_config.get('build_type', 'Release')}"
            ]

            if sys.platform.startswith('darwin'):
                macos_config = self.cmake_config.get('macos', {})
                if 'deployment_target' in macos_config:
                    args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={macos_config['deployment_target']}")
                if 'architectures' in macos_config:
                    args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(macos_config['architectures'])}")

            if sys.platform.startswith('win32'):
                windows_config = self.cmake_config.get('windows', {})
                if 'generator_platform' in windows_config:
                    args.append(f"-A{windows_config['generator_platform']}")

            if self.cmake_config.get('use_ninja', True):
                ninja_path = shutil.which("ninja")
                if ninja_path:
                    print(f"Found Ninja at: {ninja_path}")
                    args.extend(["-GNinja", f"-DCMAKE_MAKE_PROGRAM={ninja_path}"])
                else:
                    print("Ninja not found, using default CMake generator")

            return args, source_dir

        def _get_build_env(self):
            """Configure build environment."""
            env = os.environ.copy()
            env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\""
            return env

        def _configure_cmake(self, ext, cmake_args, build_temp, env):
            cmake_args, source_dir = self._get_cmake_args(ext, os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name))),
                                                          self._find_nanobind()
                                                          )

            if not os.path.exists(source_dir):
                raise RuntimeError(f"CMake source directory {source_dir} does not exist")

            if self.cmake_config.get('use_preset', False):
                preset_name = self.cmake_config.get('preset_name', 'pip-install')
                preset_file = os.path.join(source_dir, "CMakePresets.json")
                if os.path.exists(preset_file):
                    cmake_args.extend(["--preset", preset_name])
                else:
                    print(f"Warning: CMakePresets.json not found at {preset_file}, ignoring preset configuration")

            os.makedirs(build_temp, exist_ok=True)

            print(f"Configuring CMake with source directory: {source_dir}")
            print(f"CMake arguments: {' '.join(cmake_args)}")

            subprocess.check_call(
                ["cmake", source_dir] + cmake_args,
                cwd=build_temp,
                env=env
            )

        def _build_cmake(self, build_args, build_temp):
            """Build CMake project."""
            subprocess.check_call(
                ["cmake", "--build", "."] + build_args,
                cwd=build_temp
            )

    ext_modules = [CMakeExtension("per_jsp")]
    cmdclass = {"build_ext": CMakeBuild}
else:
    ext_modules = []
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False
)