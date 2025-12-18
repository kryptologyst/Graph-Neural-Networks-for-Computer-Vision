"""Setup script for the GNN CV project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gnn-cv",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Graph Neural Networks for Computer Vision using superpixel-based graph representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gnn-cv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gnn-cv-train=scripts.train:main",
            "gnn-cv-demo=demo.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
