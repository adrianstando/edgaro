from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="EDGAR",
    version="0.0.1",
    description="Explainable imbalanceD learninG benchmARk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adrianstando/EDGAR",
    author="Adrian Stańdo",  # Optional
    author_email="adrian.j.stando@gmail.com",  # Optional
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent"
    ],
    keywords="XAI, imbalance, machine learning",
    packages=["EDGAR"] + [f"EDGAR.{item}" for item in find_packages(where="EDGAR")],
    python_requires=">=3.7, <4",
    install_requires=[
        'setuptools',
        'pandas>=1.2.5',
        'numpy>=1.20.3',
        'dalex>=1.5.0',
        'imbalanced-learn>=0.9.1',
        'matplotlib>=3.4.3',
        'openml>=0.12.2',
        'pandas-profiling>=3.3.0',
        'pytest>=7.1.3',
        'scikit-learn>=1.0.2',
        'imblearn>=0.0'
    ],
    project_urls={
        "Repository": "https://github.com/adrianstando/EDGAR"
    },
)
