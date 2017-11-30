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
from sagemaker_pyspark import (SageMakerModel, S3DataPath, EndpointCreationPolicy,
                               classpath_jars, SageMakerClients, SageMakerResourceCleanup)

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import KMeansProtobufResponseRowDeserializer


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


def test_sagemakermodel_should_use_the_right_endpoint_name():

    endpoint_name = "my-existing-endpoint-123"
    model = SageMakerModel(
        endpointInstanceType="x1.128xlarge",
        endpointInitialInstanceCount=2,
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer(),
        existingEndpointName=endpoint_name,
        modelImage="some_image",
        modelPath=S3DataPath("a", "b"),
        modelEnvironmentVariables=None,
        modelExecutionRoleARN="role",
        endpointCreationPolicy=EndpointCreationPolicy.DO_NOT_CREATE,
        sagemakerClient=SageMakerClients.create_sagemaker_client(),
        prependResultRows=False,
        namePolicy=None,
        uid="uid"
    )

    assert model.endpointName == endpoint_name


def test_sagemakermodel_passes_correct_params_to_scala():

    model_image = "model-abc-123"
    model_path = S3DataPath("my-bucket", "model-abc-123")
    role_arn = "role-789"
    endpoint_instance_type = "c4.8xlarge"

    model = SageMakerModel(
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=2,
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer(),
        modelImage=model_image,
        modelPath=model_path,
        modelEnvironmentVariables=None,
        modelExecutionRoleARN=role_arn,
        endpointCreationPolicy=EndpointCreationPolicy.DO_NOT_CREATE,
        sagemakerClient=SageMakerClients.create_sagemaker_client(),
        prependResultRows=False,
        namePolicy=None,
        uid="uid"
    )

    assert model.modelImage == model_image
    assert model.modelPath.bucket == model_path.bucket
    assert model.modelExecutionRoleARN == role_arn
    assert model.endpointInstanceType == endpoint_instance_type
    assert model.existingEndpointName is None


def test_sagemakermodel_can_be_created_from_java_obj():
    endpoint_name = "my-existing-endpoint-123"
    model = SageMakerModel(
        endpointInstanceType="x1.128xlarge",
        endpointInitialInstanceCount=2,
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer(),
        existingEndpointName=endpoint_name,
        modelImage="some_image",
        modelPath=S3DataPath("a", "b"),
        modelEnvironmentVariables=None,
        modelExecutionRoleARN="role",
        endpointCreationPolicy=EndpointCreationPolicy.DO_NOT_CREATE,
        sagemakerClient=SageMakerClients.create_sagemaker_client(),
        prependResultRows=False,
        namePolicy=None,
        uid="uid"
    )

    new_model = SageMakerModel._from_java(model._to_java())
    assert new_model.uid == model.uid


def test_sagemakermodel_can_do_resource_cleanup():
    endpoint_name = "my-existing-endpoint-123"
    model = SageMakerModel(
        endpointInstanceType="x1.128xlarge",
        endpointInitialInstanceCount=2,
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer(),
        existingEndpointName=endpoint_name,
        modelImage="some_image",
        modelPath=S3DataPath("a", "b"),
        modelEnvironmentVariables=None,
        modelExecutionRoleARN="role",
        endpointCreationPolicy=EndpointCreationPolicy.DO_NOT_CREATE,
        sagemakerClient=SageMakerClients.create_sagemaker_client(),
        prependResultRows=False,
        namePolicy=None,
        uid="uid"
    )

    sm = model.sagemakerClient
    assert sm is not None

    resource_cleanup = SageMakerResourceCleanup(sm)
    assert resource_cleanup is not None

    created_resources = model.getCreatedResources()
    assert created_resources is not None

    resource_cleanup.deleteResources(created_resources)
