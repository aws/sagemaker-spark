# Changelog

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
