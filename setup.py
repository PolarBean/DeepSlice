from distutils.core import setup
from setuptools import find_packages
from pathlib import Path



this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='DeepSlice',
    packages=find_packages(),
    version='1.0.6',
    license='GPL-3.0',
    description='A package to align histology to 3D brain atlases',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DeepSlice Team',
    package_data={'DeepSlice': ['metadata/volumes/placeholder.txt', 'metadata/config.json','metadata/weights/*.txt']},
    include_package_data=True,
    author_email='harry.carey@medisin.uio.no',
    url='https://github.com/PolarBean/DeepSlice',
    download_url='https://github.com/PolarBean/DeepSlice/archive/refs/tags/1.0.6.tar.gz',
    keywords=['histology', 'brain', 'atlas', 'alignment'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scikit-image',
        'tensorflow==1.15.0',
        'h5py==2.10.0',
        'typing',
        'pandas==1.3.5',
        'requests',
        'protobuf==3.20'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
    ],

)   