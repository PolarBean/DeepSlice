from distutils.core import setup
from setuptools import find_packages
from pathlib import Path
import subprocess, os

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# if GH action hasn’t replaced it, grab the latest tag
version = os.getenv("DEEPSLICE_VERSION", "{{VERSION_PLACEHOLDER}}")
if version.startswith("{{"):
    try:
        version = (
            subprocess
            .check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=this_directory)
            .decode()
            .strip()
        )
    except Exception:
        version = "0.0.0"

setup(
    name="DeepSlice",
    packages=find_packages(),
    version=version,
    license="GPL-3.0",
    description="A package to align histology to 3D brain atlases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepSlice Team",
    package_data={
        "DeepSlice": [
            "metadata/volumes/placeholder.txt",
            "metadata/config.json",
            "metadata/weights/*.txt",
        ]
    },
    include_package_data=True,
    author_email="harry.carey@medisin.uio.no",
    url="https://github.com/PolarBean/DeepSlice",
    download_url="https://github.com/PolarBean/DeepSlice/archive/refs/tags/{{VERSION_PLACEHOLDER}}.tar.gz",
    keywords=["histology", "brain", "atlas", "alignment"],
    install_requires=[
        "numpy",
        "scikit-learn",
        "scikit-image",
        "tensorflow",
        "h5py",
        "typing",
        "pandas",
        "requests",
        "protobuf",
        "lxml",
        "urllib3==1.26.6",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
    ],
)
