=========
CHANGELOG
=========

1.1.2dev
========

* spark/pyspark: Update AWS SDK version to 1.11.350


1.1.1
=====

* spark/pyspark: Enable ICN region support for spark SDK


1.1.0
=====

* spark/pyspark: Enable NRT region support for spark SDK


1.0.5
=====

* pyspark: SageMakerModel: Fix bugs in creating model from training job, s3 file and endpoint
* spark/pyspark: XGBoostSageMakerEstimator: Fix seed hyperparameter to use correct type (Int)


1.0.4
=====

* spark/pyspark feature: LinearLearnerEstimator: Add more hyper-parameters


1.0.3
=====

* feature: XGBoostSageMakerEstimator: Fix maxDepth hyperparameter to use correct type (Int)


1.0.2
=====

* feature: Estimators: Add wrapper class for LDA algorithm
* feature: Setup: Add module pytest-xdist to enable parallel testing
* feature: Documentation: Change jar path in index.rst
* feature: Documentation: Add instructions for s3 and s3a in README
* feature: Estimators: Remove unimplemented hyper-parameter in linear learner
* feature: Tests: Remove tests in py3.5 to speed up testing
* feature: Documentation: Add instructions for building pyspark from source in readme
* feature: Wrapper: Enable conversion from python list to scala.collection.immutable.List
* feature: Setup: add coverage to the scala build
* feature: Documentation: use SparkSession, not SparkContext in PySpark README


1.0.1
=====

* feature: Estimators: add support for Amazon FactorizationMachines algorithm
* feature: Documentation: multiple updates to README, scala docs, addition of CHANGELOG.rst file
* feature: Setup: update SBT plugins
* feature: Setup: add travis file


1.0.0
=====

* Initial commit
