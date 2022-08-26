# Changelog

## v1.4.4 (2022-08-26)

### Bug Fixes and Other Changes

 * Upgrade sbt version to 1.7.1

## v1.4.3 (2022-08-23)

### Bug Fixes and Other Changes

 * Upgrade Spark and PySpark version to 3.3.0

## v1.4.2 (2021-04-23)

### Bug Fixes and Other Changes

 * upgrade pyspark

## v1.4.1 (2020-09-14)

### Bug Fixes and Other Changes

 * upgrade jquery to version 1.9.0

## v1.4.0 (2020-08-10)

### Features

 * add support for af-south-1 and eu-south-1

## v1.3.2.post0 (2020-07-23)

### Testing and Release Infrastructure

 * consolidate dependency lists

## v1.3.2 (2020-07-21)

### Bug Fixes and Other Changes

 * Update README.md

## v1.3.1.post1 (2020-06-18)

### Documentation Changes

 * update latest compatible EMR version

## v1.3.1.post0 (2020-05-25)

### Documentation Changes

 * update documentation about using sagemaker-pyspark with EMR

## v1.3.1 (2020-04-21)

### Bug Fixes and Other Changes

 * update to pyspark 2.4.5

## v1.3.0 (2020-03-31)

### Features

 * add support for cn-north-1 and cn-northwest-1

### Bug Fixes and Other Changes

 * Upgrade pyspark version to 2.3.4

## v1.2.8.post1 (2020-03-26)

### Testing and Release Infrastructure

 * correctly pipe credentials for Maven release

## v1.2.8.post0 (2020-03-25)

### Testing and Release Infrastructure

 * correctly pipe credentials for Maven publish

## v1.2.8 (2020-02-04)

### Bug Fixes and Other Changes

 * fix has-matching-changes to reflect latest script changes

## v1.2.7 (2019-12-12)

### Bug fixes and other changes

 * Migrate sonatype endpoint

## v1.2.6 (2019-10-22)

### Bug fixes and other changes

 * add 1p algorithm support for me-south-1

## v1.2.5 (2019-08-22)

### Bug fixes and other changes

 * add region support to HKG/GRU/CDG/ARN
 * update AWS Java SDK to 1.11.613

## v1.2.4 (2019-05-06)

### Bug fixes and other changes

 * freeze pyspark version to 2.3.2

## v1.2.3 (2019-04-30)

### Bug fixes and other changes

 * add --ignore-reuse-error to pypi publish

## v1.2.2.post0 (2019-04-24)

### Documentation changes

 * fix version handling in docbuild

## v1.2.2 (2019-04-12)

### Bug fixes and other changes

 * setup for automated release builds
 * set version of pyspark setup.py
 * update for new ci environment

## v1.2.1 (2018-11-05)

* encode DenseMatrix and SparseMatrix in probobuf Record format
* add new region support for BOM/SIN/LHR/YUL/SFO

## v1.2.0

* add support for GovCloud

## v1.1.4

* Increase default timeout for inference requests

## v1.1.3

* LinearLearnerEstimator: Add multi-class classifier

## v1.1.2

* Enable FRA and SYD region support for spark SDK

## v1.1.1

* Enable ICN region support for spark SDK

## v1.1.0

* Enable NRT region support for spark SDK

## v1.0.5

* SageMakerModel: Fix bugs in creating model from training job, s3 file and endpoint
* XGBoostSageMakerEstimator: Fix seed hyperparameter to use correct type (Int)

## v1.0.4

* LinearLearnerEstimator: Add more hyper-parameters


## v1.0.3

* XGBoostSageMakerEstimator: Fix maxDepth hyperparameter to use correct type (Int)

## v1.0.2

* Estimators: Add wrapper class for LDA algorithm
* Setup: Add module pytest-xdist to enable parallel testing
* Documentation: Change jar path in index.rst
* Documentation: Add instructions for s3 and s3a in README
* Estimators: Remove unimplemented hyper-parameter in linear learner
* Tests: Remove tests in py3.5 to speed up testing
* Documentation: Add instructions for building pyspark from source in readme
* Wrapper: Enable conversion from python list to scala.collection.immutable.List
* Setup: add coverage to the scala build
* Documentation: use SparkSession, not SparkContext in PySpark README


## v1.0.1

* Estimators: add support for Amazon FactorizationMachines algorithm
* Documentation: multiple updates to README, scala docs, addition of CHANGELOG.rst file
* Setup: update SBT plugins
* Setup: add travis file


## v1.0.0

* Initial commit
