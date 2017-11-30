import os
import pytest

from pyspark import SparkConf, SparkContext

from sagemaker_pyspark import (S3DataPath, EndpointCreationPolicy, RandomNamePolicyFactory,
                               SageMakerClients, IAMRole, classpath_jars)
from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator

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


def get_kmeans_estimator():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = KMeansSageMakerEstimator(
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


def test_can_create_kmeans_estimator_from_config_role():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = KMeansSageMakerEstimator(
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


def test_kmeans_has_correct_defaults():
    estimator = get_kmeans_estimator()
    assert estimator.trainingSparkDataFormat == "sagemaker"


def test_kmeansSageMakerEstimator_passes_correct_params_to_scala():

    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "kmeans-training"
    output_prefix = "kmeans-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = KMeansSageMakerEstimator(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=KMeansProtobufResponseRowDeserializer(),
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


def test_kmeansSageMakerEstimator_validates_k():

    estimator = get_kmeans_estimator()

    estimator.setK(2)
    assert estimator.getK() == 2

    # passing 0 should fail
    with pytest.raises(ValueError):
        estimator.setK(0)

    # passing a string should fail
    with pytest.raises(TypeError):
        estimator.setK("0")

    estimator._transfer_params_to_java()
    assert estimator.getK() == estimator._call_java("getK")


def test_kmeansSageMakerEstimator_validates_max_iter():

    estimator = get_kmeans_estimator()
    estimator.setMaxIter(3)
    assert estimator.getMaxIter() == 3

    with pytest.raises(ValueError):
        estimator.setMaxIter(0)

    estimator._transfer_params_to_java()
    assert estimator.getMaxIter() == estimator._call_java("getMaxIter")


def test_kmeansSageMakerEstimator_validates_tolerance():

    estimator = get_kmeans_estimator()
    estimator.setTol(0.4)
    assert estimator.getTol() == 0.4

    with pytest.raises(ValueError):
        estimator.setTol(1.2)

    with pytest.raises(ValueError):
        estimator.setTol(2)

    estimator._transfer_params_to_java()
    assert estimator.getTol() == estimator._call_java("getTol")


def test_kmeansSageMakerEstimator_validates_local_init_method():

    estimator = get_kmeans_estimator()
    estimator.setLocalInitMethod("random")

    assert estimator.getLocalInitMethod() == "random"

    with pytest.raises(ValueError):
        estimator.setLocalInitMethod("some-value")

    with pytest.raises(ValueError):
        estimator.setLocalInitMethod(1)

    estimator.setLocalInitMethod("kmeans++")
    estimator._transfer_params_to_java()
    assert estimator.getLocalInitMethod() == estimator._call_java("getLocalInitMethod")


def test_kmeansSageMakerEstimator_validates_halflife_time():

    estimator = get_kmeans_estimator()
    estimator.setHalflifeTime(3)
    assert estimator.getHalflifeTime() == 3

    estimator.setHalflifeTime(0)
    assert estimator.getHalflifeTime() == 0

    with pytest.raises(ValueError):
        estimator.setHalflifeTime(-1)

    estimator._transfer_params_to_java()
    assert estimator.getHalflifeTime() == estimator._call_java("getHalflifeTime")


def test_kmeansSageMakerEstimator_validates_epochs():

    estimator = get_kmeans_estimator()
    estimator.setEpochs(3)
    assert estimator.getEpochs() == 3

    with pytest.raises(ValueError):
        estimator.setEpochs(0)

    estimator._transfer_params_to_java()
    assert estimator.getEpochs() == estimator._call_java("getEpochs")


def test_kmeansSageMakerEstimator_validates_init_method():

    estimator = get_kmeans_estimator()
    estimator.setInitMethod("random")

    assert estimator.getInitMethod() == "random"

    with pytest.raises(ValueError):
        estimator.setInitMethod("some-value")

    with pytest.raises(ValueError):
        estimator.setInitMethod(1)

    estimator.setInitMethod("kmeans++")
    estimator._transfer_params_to_java()
    assert estimator.getInitMethod() == estimator._call_java("getInitMethod")


def test_kmeansSageMakerEstimator_validates_center_factor():

    estimator = get_kmeans_estimator()
    estimator.setCenterFactor("auto")

    assert estimator.getCenterFactor() == "auto"

    estimator.setCenterFactor(5)

    assert estimator.getCenterFactor() == "5"

    with pytest.raises(ValueError):
        estimator.setCenterFactor("some-value")

    with pytest.raises(ValueError):
        estimator.setCenterFactor(-1)

    with pytest.raises(ValueError):
        estimator.setCenterFactor("0")

    estimator.setCenterFactor(2)
    estimator._transfer_params_to_java()

    assert estimator.getCenterFactor() == estimator._call_java("getCenterFactor")


def test_kmeansSageMakerEstimator_validates_trial_num():

    estimator = get_kmeans_estimator()
    estimator.setTrialNum("auto")

    assert estimator.getTrialNum() == "auto"

    estimator.setTrialNum(5)

    assert estimator.getTrialNum() == "5"

    with pytest.raises(ValueError):
        estimator.setTrialNum("some-value")

    with pytest.raises(ValueError):
        estimator.setTrialNum(0)

    with pytest.raises(ValueError):
        estimator.setTrialNum("-1")

    estimator.setTrialNum(2)
    estimator._transfer_params_to_java()
    assert estimator.getTrialNum() == estimator._call_java("getTrialNum")


def test_kmeansSageMakerEstimator_validates_mini_batch_size():

    estimator = get_kmeans_estimator()

    estimator.setMiniBatchSize(1)
    assert estimator.getMiniBatchSize() == 1

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(0)

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(-1)

    estimator.setMiniBatchSize(100)
    estimator._transfer_params_to_java()
    assert estimator.getMiniBatchSize() == estimator._call_java("getMiniBatchSize")


def test_kmeansSageMakerEstimator_validates_feature_dim():

    estimator = get_kmeans_estimator()

    estimator.setFeatureDim(1)
    assert estimator.getFeatureDim() == 1

    with pytest.raises(ValueError):
        estimator.setFeatureDim(0)

    with pytest.raises(ValueError):
        estimator.setFeatureDim(-1)

    estimator._transfer_params_to_java()
    assert estimator.getFeatureDim() == estimator._call_java("getFeatureDim")


def test_kmeansSageMakerEstimator_validates_eval_metrics():

    estimator = get_kmeans_estimator()

    estimator.setEvalMetrics("ssd, msd")
    assert estimator.getEvalMetrics() == "ssd, msd"

    with pytest.raises(ValueError):
        estimator.setEvalMetrics("ssd, msd, some-value")

    estimator.setEvalMetrics("msd, ssd")
    estimator._transfer_params_to_java()
    assert estimator.getEvalMetrics() == estimator._call_java("getEvalMetrics")
