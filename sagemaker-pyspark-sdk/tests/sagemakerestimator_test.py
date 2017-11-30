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

from sagemaker_pyspark import (classpath_jars, SageMakerEstimator)
from sagemaker_pyspark.transformation.deserializers import KMeansProtobufResponseRowDeserializer
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


def test_sagemakerestimator_passes_correct_params_to_scala():
    training_image = "train-abc-123"
    model_image = "model-abc-123"
    training_instance_count = 2
    training_instance_type = "train-abc-123"
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 2

    estimator = SageMakerEstimator(
        trainingImage=training_image,
        modelImage=model_image,
        trainingInstanceCount=training_instance_count,
        trainingInstanceType=training_instance_type,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer()
    )

    assert estimator.trainingImage == training_image
    assert estimator.modelImage == model_image
    assert estimator.trainingInstanceType == training_instance_type
    assert estimator.trainingInstanceCount == training_instance_count
    assert estimator.endpointInitialInstanceCount == endpoint_initial_instance_count
    assert estimator.endpointInstanceType == endpoint_instance_type


def test_sagemakerestimator_default_params():
    training_image = "train-abc-123"
    model_image = "model-abc-123"
    training_instance_count = 2
    training_instance_type = "train-abc-123"
    endpoint_instance_type = "endpoint-abc-123"
    endpoint_initial_instance_count = 2

    estimator = SageMakerEstimator(
        trainingImage=training_image,
        modelImage=model_image,
        trainingInstanceCount=training_instance_count,
        trainingInstanceType=training_instance_type,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer()
    )

    assert estimator.trainingInstanceVolumeSizeInGB == 1024
    assert estimator.trainingProjectedColumns is None
    assert estimator.trainingChannelName == "train"
    assert estimator.trainingContentType is None
    assert estimator.trainingS3DataDistribution == "ShardedByS3Key"
    assert estimator.trainingSparkDataFormat == "sagemaker"
    assert estimator.trainingInputMode == "File"
    assert estimator.trainingCompressionCodec is None
    assert estimator.trainingMaxRuntimeInSeconds == 24 * 60 * 60
    assert estimator.trainingKmsKeyId is None
    assert estimator.modelPrependInputRowsToTransformationRows is True
    assert estimator.deleteStagingDataAfterTraining is True
