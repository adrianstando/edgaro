from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="edgaro",
    version="1.0.0",
    description="Explainable imbalanceD learninG compARatOr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adrianstando/edgaro",
    author="Adrian Stańdo",
    author_email="adrian.j.stando@gmail.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent"
    ],
    keywords="XAI, imbalance, machine learning, AI",
    packages=["edgaro"] + [f"edgaro.{item}" for item in find_packages(where="edgaro")],
    python_requires=">=3.8, <4",
    install_requires=[
        'setuptools',
        'pandas>=1.4.4',
        'numpy>=1.20.3',
        'dalex>=1.5.0',
        'imbalanced-learn>=0.10.1',
        'matplotlib>=3.4.3',
        'openml>=0.12.2',
        'pandas-profiling>=3.3.0',
        'pytest>=7.1.3',
        'scikit-learn>=1.1.0',
        'imblearn>=0.0',
        'xgboost>=1.5.0',
        'scipy>=1.7.3',
        'statsmodels>=0.13.2'
    ],
    project_urls={
        "Documentation": "https://adrianstando.github.io/edgaro",
        "Code repository": "https://github.com/adrianstando/edgaro"
    },
)
