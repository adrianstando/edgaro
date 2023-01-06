# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased-official]

### The first release of the package

This is the first official release of the package.

It allows to perform all required computations for comparing and benchmarking balancing methods.

## [Unreleased] - 2023-01-06

### Added

- Parallel computing support in `explain` module
- Variable Importance option in `Explainer` and `ExplainerArray` classes
- Variable Importance explanation in `ModelPartsExplanation` and `ModelPartsExplanationArray` classes in `explain` module
- Variable Importance comparison metric based on Wilcoxon statistical test
- Variable Importance plots and summary plots

### Changed

- Parameter name `curve_type` in `Explainer` and `ExplainerArray` classes changed to `explanation_type`
- `ExplainerResult` and `ExplainerResultArray` classes are changed to `ModelProfileExplanation` and `ModelProfileExplanationArray`

## [0.2.0] - 2023-01-02

### Added

- Compare method for `ExplainerResultArray`
- New plot method to summary benchmarking
- Enable compare methods to return raw values
- Benchmarking set is also available with only continuous variables

### Removed

- Two comparison metrics - only one, based on variance, is left

## [0.2.0] - 2022-12-21

### Added

- Example dataset
- Benchmarking set
- Extracted `requirements.txt` file for environment setup without using PyPI
- User manual in documentation

## [0.1.0] - 2022-12-06

### Added

- All main components are implemented
- Finished unit tests
- Finished CI/CD pipeline

