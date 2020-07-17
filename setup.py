"""
Copyright 2018 SuperDARN Canada, University of Saskatchewan

setup.py
2018-11-05
To setup pyDARN as a third party library. Include installing need libraries for
running the files.

author:
Marina Schmidt
"""

from setuptools import setup, find_packages
from os import path
import sys
from subprocess import check_call
from setuptools.command.install import install, orig


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Setup information
setup(
    name="pyDARNio",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="Python library for reading and writing SuperDARN data",
    url='https://github.com/SuperDARN/pyDARNio.git',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs', 'test']),
    author="SuperDARN",
    # used to import the logging config file into pydarn.
    include_package_data=True,
    setup_requires=['pyyaml', 'numpy',
                    'h5py', 'deepdish', 'pathlib2'],
    # pyyaml library install
    install_requires=['pyyaml', 'numpy',
                      'h5py', 'deepdish', 'pathlib2']
)
