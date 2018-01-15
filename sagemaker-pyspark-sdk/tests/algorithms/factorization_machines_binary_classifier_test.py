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
from sagemaker_pyspark.algorithms import FactorizationMachinesBinaryClassifier

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers \
    import FactorizationMachinesBinaryClassifierDeserializer


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


def get_factorization_machines_binary_classifier():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = FactorizationMachinesBinaryClassifier(
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


def test_can_create_classifier_from_configured_iam_role():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = FactorizationMachinesBinaryClassifier(
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


def test_factorization_machines_binary_classifier_has_correct_defaults():
    estimator = get_factorization_machines_binary_classifier()
    assert estimator.trainingSparkDataFormat == "sagemaker"


def test_factorizationMachinesBinaryClassifier_passes_correct_params_to_scala():

    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "factorization-machines-binary-classifier-training"
    output_prefix = "factorization-machines-binary-classifier-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = FactorizationMachinesBinaryClassifier(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=FactorizationMachinesBinaryClassifierDeserializer(),
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


def test_factorizationMachinesBinaryClassifier_validates_feature_dim():

    estimator = get_factorization_machines_binary_classifier()

    estimator.setFeatureDim(2)
    assert estimator.getFeatureDim() == 2

    # passing 0 should fail
    with pytest.raises(ValueError):
        estimator.setFeatureDim(0)

    # passing a string should fail
    with pytest.raises(TypeError):
        estimator.setFeatureDim("0")

    estimator.setFeatureDim(2)
    estimator._transfer_params_to_java()
    assert estimator.getFeatureDim() == estimator._call_java("getFeatureDim")


def test_factorizationMachinesBinaryClassifier_validates_num_factors():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setNumFactors(3)
    assert estimator.getNumFactors() == 3

    with pytest.raises(ValueError):
        estimator.setNumFactors(0)

    estimator.setNumFactors(3)
    estimator._transfer_params_to_java()
    assert estimator.getNumFactors() == estimator._call_java("getNumFactors")


def test_factorizationMachinesBinaryClassifier_validates_mini_batch_size():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setMiniBatchSize(3)
    assert estimator.getMiniBatchSize() == 3

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(0)

    estimator.setMiniBatchSize(3)
    estimator._transfer_params_to_java()
    assert estimator.getMiniBatchSize() == estimator._call_java("getMiniBatchSize")


def test_factorizationMachinesBinaryClassifier_validates_epochs():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setEpochs(3)
    assert estimator.getEpochs() == 3

    with pytest.raises(ValueError):
        estimator.setEpochs(-1)

    estimator.setEpochs(2)
    estimator._transfer_params_to_java()
    assert estimator.getEpochs() == estimator._call_java("getEpochs")


def test_factorizationMachinesBinaryClassifier_validates_bias_lr():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setBiasLr(3)
    assert estimator.getBiasLr() == 3

    with pytest.raises(ValueError):
        estimator.setBiasLr(-1)

    estimator.setBiasLr(2)
    estimator._transfer_params_to_java()
    assert estimator.getBiasLr() == estimator._call_java("getBiasLr")


def test_factorizationMachinesBinaryClassifier_validates_linear_lr():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setLinearLr(3)
    assert estimator.getLinearLr() == 3

    with pytest.raises(ValueError):
        estimator.setLinearLr(-1)

    estimator.setBiasLr(2)
    estimator._transfer_params_to_java()
    assert estimator.getLinearLr() == estimator._call_java("getLinearLr")


def test_factorizationMachinesBinaryClassifier_validates_factors_lr():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setFactorsLr(3)
    assert estimator.getFactorsLr() == 3

    with pytest.raises(ValueError):
        estimator.setFactorsLr(-1)

    estimator.setFactorsLr(2)
    estimator._transfer_params_to_java()
    assert estimator.getFactorsLr() == estimator._call_java("getFactorsLr")


def test_factorizationMachinesBinaryClassifier_validates_bias_wd():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setBiasWd(3)
    assert estimator.getBiasWd() == 3

    with pytest.raises(ValueError):
        estimator.setBiasWd(-1)

    estimator.setBiasWd(2)
    estimator._transfer_params_to_java()
    assert estimator.getBiasWd() == estimator._call_java("getBiasWd")


def test_factorizationMachinesBinaryClassifier_validates_linear_wd():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setLinearWd(3)
    assert estimator.getLinearWd() == 3

    with pytest.raises(ValueError):
        estimator.setLinearWd(-1)

    estimator.setLinearWd(2)
    estimator._transfer_params_to_java()
    assert estimator.getLinearWd() == estimator._call_java("getLinearWd")


def test_factorizationMachinesBinaryClassifier_validates_factors_wd():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setFactorsWd(3)
    assert estimator.getFactorsWd() == 3

    with pytest.raises(ValueError):
        estimator.setFactorsWd(-1)

    estimator.setFactorsWd(2)
    estimator._transfer_params_to_java()
    assert estimator.getFactorsWd() == estimator._call_java("getFactorsWd")


def test_factorizationMachinesBinaryClassifier_validates_bias_init_method():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setBiasInitMethod("uniform")
    assert estimator.getBiasInitMethod() == "uniform"

    with pytest.raises(ValueError):
        estimator.setBiasInitMethod("some-value")

    estimator.setBiasInitMethod("normal")
    estimator._transfer_params_to_java()
    assert estimator.getBiasInitMethod() == estimator._call_java("getBiasInitMethod")


def test_factorizationMachinesBinaryClassifier_validates_bias_init_scale():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setBiasInitScale(3)
    assert estimator.getBiasInitScale() == 3

    with pytest.raises(ValueError):
        estimator.setBiasInitScale(-1)

    estimator.setBiasInitScale(2)
    estimator._transfer_params_to_java()
    assert estimator.getBiasInitScale() == estimator._call_java("getBiasInitScale")


def test_factorizationMachinesBinaryClassifier_validates_bias_init_sigma():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setBiasInitSigma(3)
    assert estimator.getBiasInitSigma() == 3

    with pytest.raises(ValueError):
        estimator.setBiasInitSigma(-1)

    estimator.setBiasInitSigma(2)
    estimator._transfer_params_to_java()
    assert estimator.getBiasInitSigma() == estimator._call_java("getBiasInitSigma")


def test_factorizationMachinesBinaryClassifier_validates_linear_init_method():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setLinearInitMethod("uniform")
    assert estimator.getLinearInitMethod() == "uniform"

    with pytest.raises(ValueError):
        estimator.setLinearInitMethod("some-value")

    estimator.setLinearInitMethod("constant")
    estimator._transfer_params_to_java()
    assert estimator.getLinearInitMethod() == estimator._call_java("getLinearInitMethod")


def test_factorizationMachinesBinaryClassifier_validates_linear_init_scale():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setLinearInitScale(3)
    assert estimator.getLinearInitScale() == 3

    with pytest.raises(ValueError):
        estimator.setLinearInitScale(-1)

    estimator.setLinearInitScale(2)
    estimator._transfer_params_to_java()
    assert estimator.getLinearInitScale() == estimator._call_java("getLinearInitScale")


def test_factorizationMachinesBinaryClassifier_validates_linear_init_sigma():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setLinearInitSigma(3)
    assert estimator.getLinearInitSigma() == 3

    with pytest.raises(ValueError):
        estimator.setLinearInitSigma(-1)

    estimator.setLinearInitSigma(2)
    estimator._transfer_params_to_java()
    assert estimator.getLinearInitSigma() == estimator._call_java("getLinearInitSigma")


def test_factorizationMachinesBinaryClassifier_validates_factors_init_method():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setFactorsInitMethod("uniform")
    assert estimator.getFactorsInitMethod() == "uniform"

    with pytest.raises(ValueError):
        estimator.setFactorsInitMethod("some-value")

    estimator.setFactorsInitMethod("normal")
    estimator._transfer_params_to_java()
    assert estimator.getFactorsInitMethod() == estimator._call_java("getFactorsInitMethod")


def test_factorizationMachinesBinaryClassifier_validates_factors_init_scale():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setFactorsInitScale(3)
    assert estimator.getFactorsInitScale() == 3

    with pytest.raises(ValueError):
        estimator.setFactorsInitScale(-1)

    estimator.setFactorsInitScale(2)
    estimator._transfer_params_to_java()
    assert estimator.getFactorsInitScale() == estimator._call_java("getFactorsInitScale")


def test_factorizationMachinesBinaryClassifier_validates_factors_init_sigma():

    estimator = get_factorization_machines_binary_classifier()
    estimator.setFactorsInitSigma(3)
    assert estimator.getFactorsInitSigma() == 3

    with pytest.raises(ValueError):
        estimator.setFactorsInitSigma(-1)

    estimator.setFactorsInitSigma(2)
    estimator._transfer_params_to_java()
    assert estimator.getFactorsInitSigma() == estimator._call_java("getFactorsInitSigma")
