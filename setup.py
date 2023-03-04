from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read()

setup(
    name="variogram_analysis",
    version="0.0.10",
    description="A tool to quantify the spatio-temporal discontinuity of sparse data",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonasdieker/variogram_analysis",
    author="Jonas Dieker",
    author_email="jonasdieker05@gmail.com",
    license="MIT",
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)