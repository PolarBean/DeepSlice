from distutils.core import setup
from setuptools import find_packages
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="DeepSlice",
    packages=find_packages(),
    version="{{VERSION_PLACEHOLDER}}",
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
        "tensorflow<=2.15.0",
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
