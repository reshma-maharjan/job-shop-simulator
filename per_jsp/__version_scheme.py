import os
from setuptools_scm import get_version

def version_scheme(version):
    if version.exact:
        return version.format_with("{tag}")
    else:
        patch = int(os.environ.get("BUILD_NUMBER", 0))
        return version.format_with(f"{{tag}}.dev{{distance}}+{patch}")

def get_version_scheme(root=".", relative_to=None, version_scheme=version_scheme):
    return get_version(
        root=root,
        relative_to=relative_to,
        version_scheme=version_scheme,
    )