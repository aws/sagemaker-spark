# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#   http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import pytest

from pyspark import SparkContext, SparkConf

from sagemaker_pyspark import (S3DataPath, EndpointCreationPolicy, RandomNamePolicyFactory,
                               SageMakerClients, IAMRole, classpath_jars)
from sagemaker_pyspark.algorithms import XGBoostSageMakerEstimator

from sagemaker_pyspark.transformation.serializers import LibSVMRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import XGBoostCSVRowDeserializer


@pytest.fixture(autouse=True)
def with_spark_context():
    os.environ['SPARK_CLASSPATH'] = ":".join(classpath_jars())
    conf = (SparkConf()
            .set("spark.driver.extraClassPath", os.environ['SPARK_CLASSPATH']))

    if SparkContext._active_spark_context is None:
        SparkContext(conf=conf)

    yield SparkContext._active_spark_context

    # TearDown
    SparkContext.stop(SparkContext._active_spark_context)


def get_xgboost_estimator():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = XGBoostSageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole("some-role"),
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    return estimator


def test_can_create_xgboost_estimator_from_config_role():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = XGBoostSageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    return estimator


def test_xgboostSageMakerEstimator_has_correct_defaults():
    estimator = get_xgboost_estimator()
    assert estimator.trainingSparkDataFormat == "libsvm"


def test_xgboostSageMakerEstimator_passes_correct_params_to_scala():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "xgboost-training"
    output_prefix = "xgboost-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = XGBoostSageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=LibSVMRequestRowSerializer(),
        responseRowDeserializer=XGBoostCSVRowDeserializer(),
        trainingInstanceVolumeSizeInGB=2048,
        trainingInputS3DataPath=S3DataPath(training_bucket, input_prefix),
        trainingOutputS3DataPath=S3DataPath(training_bucket, output_prefix),
        trainingMaxRuntimeInSeconds=1,
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM,
        sagemakerClient=SageMakerClients.create_sagemaker_client(),
        s3Client=SageMakerClients.create_s3_default_client(),
        stsClient=SageMakerClients.create_sts_default_client(),
        modelPrependInputRowsToTransformationRows=True,
        namePolicyFactory=RandomNamePolicyFactory(),
        uid="sagemaker")

    assert estimator.trainingInputS3DataPath.bucket == training_bucket
    assert estimator.trainingInputS3DataPath.objectPath == input_prefix
    assert estimator.trainingInstanceCount == training_instance_count
    assert estimator.trainingInstanceType == training_instance_type
    assert estimator.endpointInstanceType == endpoint_instance_type
    assert estimator.endpointInitialInstanceCount == endpoint_initial_instance_count
    assert estimator.trainingInstanceVolumeSizeInGB == 2048
    assert estimator.trainingMaxRuntimeInSeconds == 1
    assert estimator.trainingKmsKeyId is None


def test_xgboostSageMakerEstimator_validates_booster():
    estimator = get_xgboost_estimator()

    estimator.setBooster("gbtree")
    assert estimator.getBooster() == "gbtree"

    with pytest.raises(ValueError):
        estimator.setBooster("some-random-value")

    estimator.setBooster("gblinear")
    estimator._transfer_params_to_java()
    assert estimator.getBooster() == estimator._call_java("getBooster")

    estimator.setBooster("dart")
    assert estimator.getBooster() == "dart"


def test_xgboostSageMakerEstimator_validates_silent():
    estimator = get_xgboost_estimator()

    estimator.setSilent(1)
    assert estimator.getSilent() == 1

    with pytest.raises(ValueError):
        estimator.setSilent(2)

    estimator._transfer_params_to_java()
    assert estimator.getSilent() == estimator._call_java("getSilent")

    estimator.setSilent(0)
    assert estimator.getSilent() == 0


def test_xgboostSageMakerEstimator_validates_nthread():
    estimator = get_xgboost_estimator()

    estimator.setNThread(1)
    assert estimator.getNThread() == 1

    with pytest.raises(ValueError):
        estimator.setNThread(0)

    estimator.setNThread(100)
    estimator._transfer_params_to_java()
    assert estimator.getNThread() == estimator._call_java("getNThread")


def test_xgboostSageMakerEstimator_validates_eta():
    estimator = get_xgboost_estimator()

    estimator.setEta(0.1)
    assert estimator.getEta() == 0.1

    with pytest.raises(ValueError):
        estimator.setEta(2)

    with pytest.raises(ValueError):
        estimator.setEta(-1)

    estimator._transfer_params_to_java()
    assert estimator.getEta() == estimator._call_java("getEta")

    estimator.setEta(0)
    assert estimator.getEta() == 0


def test_xgboostSageMakerEstimator_validates_gamma():
    estimator = get_xgboost_estimator()

    estimator.setGamma(1.44)
    assert estimator.getGamma() == 1.44

    with pytest.raises(ValueError):
        estimator.setGamma(-1)

    estimator._transfer_params_to_java()
    assert estimator.getGamma() == estimator._call_java("getGamma")

    estimator.setGamma(0)
    assert estimator.getGamma() == 0


def test_xgboostSageMakerEstimator_validates_maxDepth():
    estimator = get_xgboost_estimator()

    estimator.setMaxDepth(1.1)
    assert estimator.getMaxDepth() == 1.1

    with pytest.raises(ValueError):
        estimator.setMaxDepth(-1)

    estimator._transfer_params_to_java()
    assert estimator.getMaxDepth() == estimator._call_java("getMaxDepth")

    estimator.setMaxDepth(0)
    assert estimator.getMaxDepth() == 0


def test_xgboostSageMakerEstimator_validates_min_child_weight():
    estimator = get_xgboost_estimator()

    estimator.setMinChildWeight(3.14)
    assert estimator.getMinChildWeight() == 3.14

    with pytest.raises(ValueError):
        estimator.setMinChildWeight(-1)

    estimator._transfer_params_to_java()
    assert estimator.getMinChildWeight() == estimator._call_java("getMinChildWeight")

    estimator.setMinChildWeight(0)
    assert estimator.getMinChildWeight() == 0


def test_xgboostSageMakerEstimator_validates_max_delta_step():
    estimator = get_xgboost_estimator()

    estimator.setMaxDeltaStep(2.21)
    assert estimator.getMaxDeltaStep() == 2.21

    with pytest.raises(ValueError):
        estimator.setMaxDeltaStep(-1)

    estimator._transfer_params_to_java()
    assert estimator.getMaxDeltaStep() == estimator._call_java("getMaxDeltaStep")

    estimator.setMaxDeltaStep(0)
    assert estimator.getMaxDeltaStep() == 0


def test_xgboostSageMakerEstimator_validates_subsample():
    estimator = get_xgboost_estimator()

    estimator.setSubsample(0.99)
    assert estimator.getSubsample() == 0.99

    with pytest.raises(ValueError):
        estimator.setSubsample(-1)

    with pytest.raises(ValueError):
        estimator.setSubsample(0)

    estimator._transfer_params_to_java()
    assert estimator.getSubsample() == estimator._call_java("getSubsample")

    estimator.setSubsample(0.01)
    assert estimator.getSubsample() == 0.01

    estimator.setSubsample(1)
    assert estimator.getSubsample() == 1


def test_xgboostSageMakerEstimator_validates_col_sample_by_tree():
    estimator = get_xgboost_estimator()

    estimator.setColSampleByTree(0.99)
    assert estimator.getColSampleByTree() == 0.99

    with pytest.raises(ValueError):
        estimator.setColSampleByTree(-1)

    with pytest.raises(ValueError):
        estimator.setColSampleByTree(0)

    estimator._transfer_params_to_java()
    assert estimator.getColSampleByTree() == estimator._call_java("getColSampleByTree")

    estimator.setColSampleByTree(0.01)
    assert estimator.getColSampleByTree() == 0.01

    estimator.setColSampleByTree(1)
    assert estimator.getColSampleByTree() == 1


def test_xgboostSageMakerEstimator_validates_col_sample_by_level():
    estimator = get_xgboost_estimator()

    estimator.setColSampleByLevel(0.99)
    assert estimator.getColSampleByLevel() == 0.99

    with pytest.raises(ValueError):
        estimator.setColSampleByLevel(-1)

    with pytest.raises(ValueError):
        estimator.setColSampleByLevel(0)

    estimator._transfer_params_to_java()
    assert estimator.getColSampleByLevel() == estimator._call_java("getColSampleByLevel")

    estimator.setColSampleByLevel(0.01)
    assert estimator.getColSampleByLevel() == 0.01

    estimator.setColSampleByLevel(1)
    assert estimator.getColSampleByLevel() == 1


def test_xgboostSageMakerEstimator_validates_alpha():
    estimator = get_xgboost_estimator()

    estimator.setAlpha(2.21)
    assert estimator.getAlpha() == 2.21

    estimator._transfer_params_to_java()
    assert estimator.getAlpha() == estimator._call_java("getAlpha")

    estimator.setAlpha(0)
    assert estimator.getAlpha() == 0


def test_xgboostSageMakerEstimator_validates_tree_method():
    estimator = get_xgboost_estimator()

    for value in ("auto", "exact", "approx", "hist"):
        estimator.setTreeMethod(value)
        assert estimator.getTreeMethod() == value

    estimator._transfer_params_to_java()
    assert estimator.getTreeMethod() == estimator._call_java("getTreeMethod")

    with pytest.raises(ValueError):
        estimator.setTreeMethod("1")

    with pytest.raises(ValueError):
        estimator.setTreeMethod("some other value")


def test_xgboostSageMakerEstimator_validates_sketch_eps():
    estimator = get_xgboost_estimator()

    estimator.setSketchEps(0.99)
    assert estimator.getSketchEps() == 0.99

    with pytest.raises(ValueError):
        estimator.setSketchEps(-1)

    with pytest.raises(ValueError):
        estimator.setSketchEps(0)

    with pytest.raises(ValueError):
        estimator.setSketchEps(1)

    estimator._transfer_params_to_java()
    assert estimator.getSketchEps() == estimator._call_java("getSketchEps")

    estimator.setSketchEps(0.01)
    assert estimator.getSketchEps() == 0.01


def test_xgboostSageMakerEstimator_validates_scale_pos_weight():
    estimator = get_xgboost_estimator()

    estimator.setScalePosWeight(1.132)
    assert estimator.getScalePosWeight() == 1.132

    estimator._transfer_params_to_java()
    assert estimator.getScalePosWeight() == estimator._call_java("getScalePosWeight")

    estimator.setScalePosWeight(0)
    assert estimator.getScalePosWeight() == 0


def test_xgboostSageMakerEstimator_validates_updater():
    estimator = get_xgboost_estimator()

    estimator.setUpdater("distcol")
    assert estimator.getUpdater() == "distcol"

    estimator.setUpdater("distcol, sync, refresh")
    assert estimator.getUpdater() == "distcol, sync, refresh"

    with pytest.raises(ValueError):
        estimator.setUpdater("invalid,updater,value")

    estimator._transfer_params_to_java()
    assert estimator.getUpdater() == estimator._call_java("getUpdater")


def test_xgboostSageMakerEstimator_validates_refresh_leaf():
    estimator = get_xgboost_estimator()

    estimator.setRefreshLeaf(0)
    assert estimator.getRefreshLeaf() == 0

    estimator.setRefreshLeaf(1)
    assert estimator.getRefreshLeaf() == 1

    with pytest.raises(ValueError):
        estimator.setRefreshLeaf(-1)

    with pytest.raises(ValueError):
        estimator.setRefreshLeaf(2)

    with pytest.raises(ValueError):
        estimator.setRefreshLeaf(0.5)

    estimator._transfer_params_to_java()
    assert estimator.getRefreshLeaf() == estimator._call_java("getRefreshLeaf")


def test_xgboostSageMakerEstimator_validates_process_type():
    estimator = get_xgboost_estimator()

    for value in ("default", "update"):
        estimator.setProcessType(value)
        assert estimator.getProcessType() == value

    estimator._transfer_params_to_java()
    assert estimator.getProcessType() == estimator._call_java("getProcessType")

    with pytest.raises(ValueError):
        estimator.setProcessType("invalid")


def test_xgboostSageMakerEstimator_validates_grow_policy():
    estimator = get_xgboost_estimator()

    for value in ("depthwise", "lossguide"):
        estimator.setGrowPolicy(value)
        assert estimator.getGrowPolicy() == value

    estimator._transfer_params_to_java()
    assert estimator.getGrowPolicy() == estimator._call_java("getGrowPolicy")

    with pytest.raises(ValueError):
        estimator.setGrowPolicy("invalid")


def test_xgboostSageMakerEstimator_validates_max_leaves():
    estimator = get_xgboost_estimator()

    estimator.setMaxLeaves(1)
    assert estimator.getMaxLeaves() == 1

    with pytest.raises(ValueError):
        estimator.setMaxLeaves(-1)

    estimator._transfer_params_to_java()
    assert estimator.getMaxLeaves() == estimator._call_java("getMaxLeaves")

    estimator.setMaxLeaves(0)
    assert estimator.getMaxLeaves() == 0


def test_xgboostSageMakerEstimator_validates_max_bin():
    estimator = get_xgboost_estimator()

    estimator.setMaxBin(1)
    assert estimator.getMaxBin() == 1

    with pytest.raises(ValueError):
        estimator.setMaxBin(-1)

    with pytest.raises(ValueError):
        estimator.setMaxBin(0)

    estimator.setMaxBin(128)
    assert estimator.getMaxBin() == 128

    estimator._transfer_params_to_java()
    assert estimator.getMaxBin() == estimator._call_java("getMaxBin")


def test_xgboostSageMakerEstimator_validates_sample_type():
    estimator = get_xgboost_estimator()

    for value in ("uniform", "weighted"):
        estimator.setSampleType(value)
        assert estimator.getSampleType() == value

    estimator._transfer_params_to_java()
    assert estimator.getSampleType() == estimator._call_java("getSampleType")

    with pytest.raises(ValueError):
        estimator.setSampleType("invalid")


def test_xgboostSageMakerEstimator_validates_normalize_type():
    estimator = get_xgboost_estimator()

    for value in ("tree", "forest"):
        estimator.setNormalizeType(value)
        assert estimator.getNormalizeType() == value

    estimator._transfer_params_to_java()
    assert estimator.getNormalizeType() == estimator._call_java("getNormalizeType")

    with pytest.raises(ValueError):
        estimator.setNormalizeType("invalid")


def test_xgboostSageMakerEstimator_validates_rate_drop():
    estimator = get_xgboost_estimator()

    estimator.setRateDrop(1)
    assert estimator.getRateDrop() == 1

    with pytest.raises(ValueError):
        estimator.setRateDrop(-1)

    estimator.setRateDrop(0)
    assert estimator.getRateDrop() == 0

    estimator.setRateDrop(0.5)
    assert estimator.getRateDrop() == 0.5

    estimator._transfer_params_to_java()
    assert estimator.getRateDrop() == estimator._call_java("getRateDrop")


def test_xgboostSageMakerEstimator_validates_one_drop():
    estimator = get_xgboost_estimator()

    estimator.setOneDrop(0)
    assert estimator.getOneDrop() == 0

    estimator.setOneDrop(1)
    assert estimator.getOneDrop() == 1

    with pytest.raises(ValueError):
        estimator.setOneDrop(-1)

    with pytest.raises(ValueError):
        estimator.setOneDrop(2)

    with pytest.raises(ValueError):
        estimator.setOneDrop(0.5)

    estimator._transfer_params_to_java()
    assert estimator.getOneDrop() == estimator._call_java("getOneDrop")


def test_xgboostSageMakerEstimator_validates_skip_drop():
    estimator = get_xgboost_estimator()

    estimator.setSkipDrop(1)
    assert estimator.getSkipDrop() == 1

    with pytest.raises(ValueError):
        estimator.setSkipDrop(-1)

    estimator.setSkipDrop(0)
    assert estimator.getSkipDrop() == 0

    estimator.setSkipDrop(0.5)
    assert estimator.getSkipDrop() == 0.5

    estimator._transfer_params_to_java()
    assert estimator.getSkipDrop() == estimator._call_java("getSkipDrop")


def test_xgboostSageMakerEstimator_validates_lambda_bias():
    estimator = get_xgboost_estimator()

    estimator.setLambdaBias(0.5)
    assert estimator.getLambdaBias() == 0.5

    with pytest.raises(ValueError):
        estimator.setLambdaBias(-1)

    with pytest.raises(ValueError):
        estimator.setLambdaBias(-3)

    estimator.setLambdaBias(0)
    assert estimator.getLambdaBias() == 0

    estimator._transfer_params_to_java()
    assert estimator.getLambdaBias() == estimator._call_java("getLambdaBias")


def test_xgboostSageMakerEstimator_validates_tweedie_variance_power():
    estimator = get_xgboost_estimator()

    estimator.setTweedieVariancePower(1.01)
    assert estimator.getTweedieVariancePower() == 1.01

    estimator.setTweedieVariancePower(1.99)
    assert estimator.getTweedieVariancePower() == 1.99

    estimator.setTweedieVariancePower(1.5)
    assert estimator.getTweedieVariancePower() == 1.5

    with pytest.raises(ValueError):
        estimator.setTweedieVariancePower(-1)

    with pytest.raises(ValueError):
        estimator.setTweedieVariancePower(1)

    with pytest.raises(ValueError):
        estimator.setTweedieVariancePower(2)

    with pytest.raises(ValueError):
        estimator.setTweedieVariancePower(2.1)

    with pytest.raises(ValueError):
        estimator.setTweedieVariancePower(0)

    estimator._transfer_params_to_java()
    assert estimator.getTweedieVariancePower() == estimator._call_java("getTweedieVariancePower")


def test_xgboostSageMakerEstimator_validates_objective():
    estimator = get_xgboost_estimator()

    for value in ("reg:linear", "reg:logistic", "binary:logistic", "binary:logistraw",
                  "count:poisson", "multi:softmax", "multi:softprob", "rank:pairwise",
                  "reg:gamma", "reg:tweedie"):
        estimator.setObjective(value)
        assert estimator.getObjective() == value

    estimator._transfer_params_to_java()
    assert estimator.getObjective() == estimator._call_java("getObjective")

    with pytest.raises(ValueError):
        estimator.setObjective("invalid")


def test_xgboostSageMakerEstimator_validates_num_classes():
    estimator = get_xgboost_estimator()

    estimator.setNumClasses(1)
    assert estimator.getNumClasses() == 1

    estimator.setNumClasses(2)
    assert estimator.getNumClasses() == 2

    estimator.setNumClasses(100)
    assert estimator.getNumClasses() == 100

    with pytest.raises(ValueError):
        estimator.setNumClasses(-1)

    with pytest.raises(ValueError):
        estimator.setNumClasses(0)

    estimator._transfer_params_to_java()
    assert estimator.getNumClasses() == estimator._call_java("getNumClasses")


def test_xgboostSageMakerEstimator_validates_base_score():
    estimator = get_xgboost_estimator()

    estimator.setBaseScore(1)
    assert estimator.getBaseScore() == 1

    estimator.setBaseScore(0)
    assert estimator.getBaseScore() == 0

    estimator.setBaseScore(0.5)
    assert estimator.getBaseScore() == 0.5

    estimator._transfer_params_to_java()
    assert estimator.getBaseScore() == estimator._call_java("getBaseScore")


def test_xgboostSageMakerEstimator_validates_eval_metric():
    estimator = get_xgboost_estimator()

    for value in ("rmse", "mae", "logloss", "error", "error@t", "merror",
                  "mlogloss", "auc", "ndcg", "map", "ndcg@n", "ndcg-", "ndcg@n-",
                  "map-", "map@n-"):
        estimator.setEvalMetric(value)
        assert estimator.getEvalMetric() == value

    estimator._transfer_params_to_java()
    assert estimator.getEvalMetric() == estimator._call_java("getEvalMetric")

    with pytest.raises(ValueError):
        estimator.setEvalMetric("invalid")


def test_xgboostSageMakerEstimator_can_set_seed():
    estimator = get_xgboost_estimator()

    estimator.setSeed(128)
    assert estimator.getSeed() == 128

    estimator._transfer_params_to_java()

    assert estimator.getSeed() == estimator._call_java("getSeed")


def test_xgboostSageMakerEstimator_validates_num_round():
    estimator = get_xgboost_estimator()

    estimator.setNumRound(1)
    assert estimator.getNumRound() == 1

    estimator.setNumRound(100)
    assert estimator.getNumRound() == 100

    with pytest.raises(ValueError):
        estimator.setNumRound(0)

    estimator._transfer_params_to_java()
    assert estimator.getNumRound() == estimator._call_java("getNumRound")
