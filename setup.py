"""
Setup script for Adaptive-Frequency-SSM
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-frequency-ssm",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Adaptive-Frequency-SSM: Advanced Frequency Domain State Space Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research/adaptive-frequency-ssm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "einops>=0.6.0",
        "datasets>=2.0.0",
        "transformers>=4.20.0",
        "wandb>=0.13.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "experiments": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.7.0",
            "plotly>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-frequency-ssm-train=adaptive_frequency_ssm.train:main",
            "adaptive-frequency-ssm-eval=adaptive_frequency_ssm.evaluation:main",
        ],
    },
    package_data={
        "adaptive_frequency_ssm": ["configs/*.yaml", "configs/*.json"],
    },
    include_package_data=True,
    zip_safe=False,
)