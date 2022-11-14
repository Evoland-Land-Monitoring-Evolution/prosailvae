#!/usr/bin/env python
"""Setup script for building prosail's python bindings"""
import os
import codecs
import re
from os import path
from setuptools import setup

# Global variables for this extension:
name = "prosailvae"  # name of the generated python extension (.so)
description = "VAE with prosail simulator"
long_description = "VAE with prosail simulator"

this_directory = path.abspath(path.dirname(__file__))


def read(filename):
    with open(os.path.join(this_directory, filename), "rb") as f:
        return f.read().decode("utf-8")


if os.path.exists("README.md"):
    long_description = read("README.md")


def read(*parts):
    with codecs.open(os.path.join(this_directory, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


author = "J Gomez-Dans/NCEO & University College London"
author_email = "yoel.zerah@univ-toulouse.fr"
url = "https://src.koda.cnrs.fr/yoel.zerah.1/prosailpython"
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Natural Language :: English', 'Operating System :: OS Independent',
    'Programming Language :: Python :: 2', 'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries :: Python Modules',
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: GIS",
    'Intended Audience :: Science/Research',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Developers', 'Environment :: Console'
]

setup(
    name=name,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=author,
    url=url,
    author_email=author_email,
    classifiers=classifiers,
    package_data={"prosailvae": ["*.txt"]},
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "pytest",
    ],
    version=find_version("prosailvae", "__init__.py"),
    packages=["prosailvae"],
    zip_safe=False  # Apparently needed for conda
)
