from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evolutionary-ml-engine",
    version="0.1.0",
    author="Harry Peter Alesso",
    author_email="your.email@example.com",
    description="Autonomous machine learning that adapts itself to changing data without human intervention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/evolutionary_ml_engine",
    py_modules=["evolutionary_ml_engine"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scipy>=1.10.0,<2.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.7.0,<4.0.0",
            "seaborn>=0.12.0,<1.0.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0,<2.0.0",
            "plotly>=5.17.0,<6.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evolutionary-ml=evolutionary_ml_engine:main",
        ],
    },
)
