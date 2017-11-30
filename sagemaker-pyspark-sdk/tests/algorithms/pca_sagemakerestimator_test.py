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
from sagemaker_pyspark.algorithms import PCASageMakerEstimator

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import PCAProtobufResponseRowDeserializer


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


def get_pca_estimator():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = PCASageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole("some-role"),
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    return estimator


def test_can_create_pca_estimator_from_config_role():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = PCASageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    return estimator


def test_PCASageMakerEstimator_has_correct_defaults():
    estimator = get_pca_estimator()
    assert estimator.trainingSparkDataFormat == "sagemaker"


def test_pcaSageMakerEstimator_passes_correct_params_to_scala():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "pca-training"
    output_prefix = "pca-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = PCASageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=PCAProtobufResponseRowDeserializer(),
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


def test_pcaSageMakerEstimator_validates_num_components():

    estimator = get_pca_estimator()

    estimator.setNumComponents(2)
    assert estimator.getNumComponents() == 2

    # passing 0 should fail
    with pytest.raises(ValueError):
        estimator.setNumComponents(0)

    # passing a string should fail
    with pytest.raises(TypeError):
        estimator.setNumComponents("0")

    estimator._transfer_params_to_java()
    assert estimator.getNumComponents() == estimator._call_java("getNumComponents")


def test_pcaSageMakerEstimator_validates_algorithm_mode():

    estimator = get_pca_estimator()
    estimator.setAlgorithmMode("regular")

    assert estimator.getAlgorithmMode() == "regular"

    with pytest.raises(ValueError):
        estimator.setAlgorithmMode("some-value")

    with pytest.raises(ValueError):
        estimator.setAlgorithmMode(1)

    estimator.setAlgorithmMode("stable")
    estimator._transfer_params_to_java()
    assert estimator.getAlgorithmMode() == estimator._call_java("getAlgorithmMode")

    estimator.setAlgorithmMode("randomized")
    assert estimator.getAlgorithmMode() == "randomized"


def test_pcaSageMakerEstimator_validates_subtract_mean():

    estimator = get_pca_estimator()
    estimator.setSubtractMean("True")

    assert estimator.getSubtractMean() is True

    with pytest.raises(ValueError):
        estimator.setSubtractMean("some-value")

    with pytest.raises(ValueError):
        estimator.setSubtractMean(1)

    estimator.setSubtractMean("False")
    estimator._transfer_params_to_java()
    assert estimator.getSubtractMean() == estimator._call_java("getSubtractMean")


def test_pcaSageMakerEstimator_validates_extra_components():

    estimator = get_pca_estimator()

    estimator.setExtraComponents(2)
    assert estimator.getExtraComponents() == 2

    # passing 0 should fail
    with pytest.raises(ValueError):
        estimator.setExtraComponents(0)

    with pytest.raises(ValueError):
        estimator.setExtraComponents(-2)

    estimator.setExtraComponents(-1)
    estimator._transfer_params_to_java()
    assert estimator.getExtraComponents() == estimator._call_java("getExtraComponents")


def test_pcaSageMakerEstimator_validates_mini_batch_size():

    estimator = get_pca_estimator()

    estimator.setMiniBatchSize(1)
    assert estimator.getMiniBatchSize() == 1

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(0)

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(-1)

    estimator.setMiniBatchSize(100)
    estimator._transfer_params_to_java()
    assert estimator.getMiniBatchSize() == estimator._call_java("getMiniBatchSize")


def test_pcaSageMakerEstimator_validates_feature_dim():

    estimator = get_pca_estimator()

    estimator.setFeatureDim(1)
    assert estimator.getFeatureDim() == 1

    with pytest.raises(ValueError):
        estimator.setFeatureDim(0)

    with pytest.raises(ValueError):
        estimator.setFeatureDim(-1)

    estimator._transfer_params_to_java()
    assert estimator.getFeatureDim() == estimator._call_java("getFeatureDim")
