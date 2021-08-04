import setuptools
import os
from setuptools import setup, find_packages
from subprocess import call


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matsml",
    version="0.0.1",
    author="Huan Tran",
    author_email="huantd@gmail.com",
    description="A machine learning toolkit for materials science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huantd/matsml.git",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    package=['matsml'],
    package_dir={'matsml': 'matsml'},
    python_requires=">=3.6",
)

