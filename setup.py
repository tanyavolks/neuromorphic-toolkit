#!/usr/bin/env python3
"""
Setup script for Neuromorphic SNN Toolkit
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "A hardware-agnostic Spiking Neural Network development toolkit"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = []
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove optional dependencies for base install
                    if "loihi-nx" not in line and "tonic" not in line:
                        requirements.append(line)
            return requirements
    except FileNotFoundError:
        return [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "numpy>=1.19.0",
            "matplotlib>=3.3.0"
        ]

setup(
    name="neuromorphic-snn-toolkit",
    version="0.1.0",
    author="Neuromorphic AI Team",
    author_email="contact@neuromorphic-ai.org",
    description="A hardware-agnostic Spiking Neural Network development toolkit",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/neuromorphic-ai/snn-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/neuromorphic-ai/snn-toolkit/issues",
        "Documentation": "https://neuromorphic-ai.github.io/snn-toolkit/",
        "Source Code": "https://github.com/neuromorphic-ai/snn-toolkit",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "full": [
            "spikingjelly>=0.0.0.0.14",
            "snntorch>=0.6.0",
            "brian2>=2.5.0",
            "loihi-nx",
            "tonic>=1.0.0"
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0"
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "snn-train=examples.mnist_classifier:main",
            "snn-demo=examples.mnist_classifier:main",
        ],
    },
    include_package_data=True,
    package_data={
        "toolkit": ["*.py"],
        "examples": ["*.py"],
        "tests": ["*.py"],
        "docs": ["*.md", "*.rst"],
    },
    keywords=[
        "spiking neural networks",
        "neuromorphic computing",
        "machine learning",
        "artificial intelligence",
        "pytorch",
        "deep learning",
        "spike trains",
        "temporal coding"
    ],
    zip_safe=False,
)