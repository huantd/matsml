from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
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
        # Replace with an actual license classifier if you use GPL:
        # "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    package_dir={'matsml': 'matsml'},  
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18",
        "pandas>=1.0",
        "scikit-learn>=0.22",
        "matplotlib",
        "scipy>=1.1.0",         
    ],
    extras_require={
        "tf": [
            "tensorflow",
            "tf-keras",
            "keras",
            "tensorflow_probability",
        ]
    },
)

