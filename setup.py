import setuptools
import os
from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matsml",
    version="1.3.0",
    author="Huan Tran",
    author_email="huantd@gmail.com",
    description="A toolkit for easy machine learning in materials science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huantd/matsml.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    package=['matsml'],
    package_dir={'matsml': 'matsml'},
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=["numpy","tensorflow","wheel","sklearn","pandas","keras","tensorflow_probability"]
)
