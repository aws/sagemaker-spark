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

from pyspark import SparkConf, SparkContext

from sagemaker_pyspark import (S3DataPath, EndpointCreationPolicy, RandomNamePolicyFactory,
                               SageMakerClients, IAMRole, classpath_jars)
from sagemaker_pyspark.algorithms import LDASageMakerEstimator

from sagemaker_pyspark.transformation.deserializers import LDAProtobufResponseRowDeserializer
from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer


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


def get_lda_estimator():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = LDASageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole("some-role"),
        trainingProjectedColumns=None,
        trainingS3DataDistribution="by-key",
        trainingInputMode="File",
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM,
        modelPrependInputRowsToTransformationRows=True,
        namePolicyFactory=RandomNamePolicyFactory(),
        uid="sagemaker")
    return estimator


def test_can_create_lda_estimator_from_config_role():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = LDASageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        trainingProjectedColumns=None,
        trainingS3DataDistribution="by-key",
        trainingInputMode="File",
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM,
        modelPrependInputRowsToTransformationRows=True,
        namePolicyFactory=RandomNamePolicyFactory(),
        uid="sagemaker")
    return estimator


def test_lda_has_correct_defaults():
    estimator = get_lda_estimator()
    assert estimator.trainingSparkDataFormat == "sagemaker"


def test_ldaSageMakerEstimator_passes_correct_params_to_scala():

    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "lda-training"
    output_prefix = "lda-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = LDASageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=LDAProtobufResponseRowDeserializer(),
        trainingInstanceVolumeSizeInGB=2048,
        trainingInputS3DataPath=S3DataPath(training_bucket, input_prefix),
        trainingOutputS3DataPath=S3DataPath(training_bucket, output_prefix),
        trainingMaxRuntimeInSeconds=1,
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM,
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


def test_ldaSageMakerEstimator_validates_num_topics():

    estimator = get_lda_estimator()

    estimator.setNumTopics(2)
    assert estimator.getNumTopics() == 2

    # passing 0 should fail
    with pytest.raises(ValueError):
        estimator.setNumTopics(0)

    # passing a string should fail
    with pytest.raises(TypeError):
        estimator.setNumTopics("0")

    estimator._transfer_params_to_java()
    assert estimator.getNumTopics() == estimator._call_java("getNumTopics")


def test_ldaSageMakerEstimator_validates_alpha0():

    estimator = get_lda_estimator()
    estimator.setAlpha0(3.3)
    assert estimator.getAlpha0() == 3.3

    with pytest.raises(ValueError):
        estimator.setAlpha0(0)

    # passing a string should fail
    with pytest.raises(TypeError):
        estimator.setAlpha0("0")

    estimator._transfer_params_to_java()
    assert estimator.getAlpha0() == estimator._call_java("getAlpha0")


def test_ldaSageMakerEstimator_validates_max_restarts():

    estimator = get_lda_estimator()
    estimator.setMaxRestarts(4)
    assert estimator.getMaxRestarts() == 4

    with pytest.raises(ValueError):
        estimator.setMaxRestarts(0)

    with pytest.raises(TypeError):
        estimator.setMaxRestarts("2")

    estimator._transfer_params_to_java()
    assert estimator.getMaxRestarts() == estimator._call_java("getMaxRestarts")


def test_ldaSageMakerEstimator_validates_max_iterations():

    estimator = get_lda_estimator()
    estimator.setMaxIterations(11)
    assert estimator.getMaxIterations() == 11

    with pytest.raises(TypeError):
        estimator.setMaxIterations("some-value")

    with pytest.raises(ValueError):
        estimator.setMaxIterations(0)

    estimator._transfer_params_to_java()
    assert estimator.getMaxIterations() == estimator._call_java("getMaxIterations")


def test_ldaSageMakerEstimator_validates_mini_batch_size():

    estimator = get_lda_estimator()

    estimator.setMiniBatchSize(1)
    assert estimator.getMiniBatchSize() == 1

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(0)

    with pytest.raises(TypeError):
        estimator.setMiniBatchSize("-1")

    estimator.setMiniBatchSize(100)
    estimator._transfer_params_to_java()
    assert estimator.getMiniBatchSize() == estimator._call_java("getMiniBatchSize")


def test_ldaSageMakerEstimator_validates_feature_dim():

    estimator = get_lda_estimator()

    estimator.setFeatureDim(1)
    assert estimator.getFeatureDim() == 1

    with pytest.raises(ValueError):
        estimator.setFeatureDim(0)

    with pytest.raises(TypeError):
        estimator.setFeatureDim("-1")

    estimator._transfer_params_to_java()
    assert estimator.getFeatureDim() == estimator._call_java("getFeatureDim")
