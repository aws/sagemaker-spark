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
from sagemaker_pyspark.algorithms import FactorizationMachinesRegressor

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers \
    import FactorizationMachinesRegressorDeserializer


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


def get_factorization_machines_regressor():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = FactorizationMachinesRegressor(
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


def test_can_create_regressor_from_configured_role():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = FactorizationMachinesRegressor(
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


def test_factorization_machines_regressor_has_correct_defaults():
    estimator = get_factorization_machines_regressor()
    assert estimator.trainingSparkDataFormat == "sagemaker"


def test_factorizationMachinesRegressor_passes_correct_params_to_scala():

    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "factorization-machines-regressor-training"
    output_prefix = "factorization-machines-regressor-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = FactorizationMachinesRegressor(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=FactorizationMachinesRegressorDeserializer(),
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

