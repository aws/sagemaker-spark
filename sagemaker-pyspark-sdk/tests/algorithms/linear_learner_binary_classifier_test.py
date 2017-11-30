import os
import pytest

from pyspark import SparkConf, SparkContext

from sagemaker_pyspark import (S3DataPath, EndpointCreationPolicy, RandomNamePolicyFactory,
                               SageMakerClients, IAMRole, classpath_jars)
from sagemaker_pyspark.algorithms import LinearLearnerBinaryClassifier

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers \
    import LinearLearnerBinaryClassifierProtobufResponseRowDeserializer


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


def get_linear_learner_binary_classifier():
    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3
    estimator = LinearLearnerBinaryClassifier(
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
    estimator = LinearLearnerBinaryClassifier(
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


def test_linear_learner_binary_classifier_has_correct_defaults():
    estimator = get_linear_learner_binary_classifier()
    assert estimator.trainingSparkDataFormat == "sagemaker"


def test_linearLearnerBinaryClassifier_passes_correct_params_to_scala():

    training_instance_type = "c4.8xlarge"
    training_instance_count = 3
    endpoint_instance_type = "c4.8xlarge"
    endpoint_initial_instance_count = 3

    training_bucket = "random-bucket"
    input_prefix = "linear-learner-binary-classifier-training"
    output_prefix = "linear-learner-binary-classifier-out"
    integTestingRole = "arn:aws:iam::123456789:role/SageMakerRole"

    estimator = LinearLearnerBinaryClassifier(
        trainingInstanceType=training_instance_type,
        trainingInstanceCount=training_instance_count,
        endpointInstanceType=endpoint_instance_type,
        endpointInitialInstanceCount=endpoint_initial_instance_count,
        sagemakerRole=IAMRole(integTestingRole),
        requestRowSerializer=ProtobufRequestRowSerializer(),
        responseRowDeserializer=LinearLearnerBinaryClassifierProtobufResponseRowDeserializer(),
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


def test_linearLearnerBinaryClassifier_validates_feature_dim():

    estimator = get_linear_learner_binary_classifier()

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


def test_linearLearnerBinaryClassifier_validates_mini_batch_size():

    estimator = get_linear_learner_binary_classifier()
    estimator.setMiniBatchSize(3)
    assert estimator.getMiniBatchSize() == 3

    with pytest.raises(ValueError):
        estimator.setMiniBatchSize(0)

    estimator.setMiniBatchSize(3)
    estimator._transfer_params_to_java()
    assert estimator.getMiniBatchSize() == estimator._call_java("getMiniBatchSize")


def test_linearLearnerBinaryClassifier_validates_epochs():

    estimator = get_linear_learner_binary_classifier()
    estimator.setEpochs(3)
    assert estimator.getEpochs() == 3

    with pytest.raises(ValueError):
        estimator.setEpochs(-1)

    estimator.setEpochs(2)
    estimator._transfer_params_to_java()
    assert estimator.getEpochs() == estimator._call_java("getEpochs")


def test_linearLearnerBinaryClassifier_validates_use_bias():

    estimator = get_linear_learner_binary_classifier()
    estimator.setUseBias("True")
    assert estimator.getUseBias() is True

    with pytest.raises(ValueError):
        estimator.setUseBias("some-value")

    estimator.setUseBias("False")
    estimator._transfer_params_to_java()
    assert estimator.getUseBias() == estimator._call_java("getUseBias")


def test_linearLearnerBinaryClassifier_validates_num_models():

    estimator = get_linear_learner_binary_classifier()
    estimator.setNumModels(3)
    assert estimator.getNumModels() == "3"

    estimator.setNumModels("auto")
    assert estimator.getNumModels() == "auto"

    with pytest.raises(ValueError):
        estimator.setNumModels(-1)

    with pytest.raises(ValueError):
        estimator.setNumModels("some-value")

    estimator.setNumModels("2")
    estimator._transfer_params_to_java()
    assert estimator.getNumModels() == estimator._call_java("getNumModels")


def test_linearLearnerBinaryClassifier_validates_num_calibration_samples():

    estimator = get_linear_learner_binary_classifier()
    estimator.setNumCalibrationSamples(3)
    assert estimator.getNumCalibrationSamples() == 3

    with pytest.raises(ValueError):
        estimator.setNumCalibrationSamples(-1)

    estimator.setNumCalibrationSamples(2)
    estimator._transfer_params_to_java()
    assert estimator.getNumCalibrationSamples() == estimator._call_java("getNumCalibrationSamples")


def test_linearLearnerBinaryClassifier_validates_init_method():

    estimator = get_linear_learner_binary_classifier()
    estimator.setInitMethod("uniform")
    assert estimator.getInitMethod() == "uniform"

    with pytest.raises(ValueError):
        estimator.setInitMethod("some-value")

    estimator.setInitMethod("normal")
    estimator._transfer_params_to_java()
    assert estimator.getInitMethod() == estimator._call_java("getInitMethod")


def test_linearLearnerBinaryClassifier_validates_init_scale():

    estimator = get_linear_learner_binary_classifier()
    estimator.setInitScale(1)
    assert estimator.getInitScale() == 1

    with pytest.raises(ValueError):
        estimator.setInitScale(0)

    estimator.setInitScale(0.5)
    estimator._transfer_params_to_java()
    assert estimator.getInitScale() == estimator._call_java("getInitScale")


def test_linearLearnerBinaryClassifier_validates_init_sigma():

    estimator = get_linear_learner_binary_classifier()
    estimator.setInitSigma(0.5)
    assert estimator.getInitSigma() == 0.5

    with pytest.raises(ValueError):
        estimator.setInitSigma(0)

    estimator.setInitSigma(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getInitSigma() == estimator._call_java("getInitSigma")


def test_linearLearnerBinaryClassifier_validates_optimizer():

    estimator = get_linear_learner_binary_classifier()
    estimator.setOptimizer("adam")
    assert estimator.getOptimizer() == "adam"

    with pytest.raises(ValueError):
        estimator.setOptimizer("some-value")

    estimator.setOptimizer("sgd")
    estimator._transfer_params_to_java()
    assert estimator.getOptimizer() == estimator._call_java("getOptimizer")


def test_linearLearnerBinaryClassifier_validates_loss():

    estimator = get_linear_learner_binary_classifier()
    estimator.setLoss("auto")
    assert estimator.getLoss() == "auto"

    with pytest.raises(ValueError):
        estimator.setLoss("some-value")

    estimator.setLoss("logistic")
    estimator._transfer_params_to_java()
    assert estimator.getLoss() == estimator._call_java("getLoss")


def test_linearLearnerBinaryClassifier_validates_wd():

    estimator = get_linear_learner_binary_classifier()
    estimator.setWd(0.5)
    assert estimator.getWd() == 0.5

    with pytest.raises(ValueError):
        estimator.setWd(-1)

    estimator.setWd(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getWd() == estimator._call_java("getWd")


def test_linearLearnerBinaryClassifier_validates_l1():

    estimator = get_linear_learner_binary_classifier()
    estimator.setL1(0.5)
    assert estimator.getL1() == 0.5

    with pytest.raises(ValueError):
        estimator.setL1(-1)

    estimator.setL1(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getL1() == estimator._call_java("getL1")


def test_linearLearnerBinaryClassifier_validates_momentum():

    estimator = get_linear_learner_binary_classifier()
    estimator.setMomentum(0.5)
    assert estimator.getMomentum() == 0.5

    with pytest.raises(ValueError):
        estimator.setMomentum(3)

    estimator.setMomentum(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getMomentum() == estimator._call_java("getMomentum")


def test_linearLearnerBinaryClassifier_validates_learning_rate():

    estimator = get_linear_learner_binary_classifier()
    estimator.setLearningRate(0.5)
    assert estimator.getLearningRate() == "0.5"

    estimator.setLearningRate("auto")
    assert estimator.getLearningRate() == "auto"

    with pytest.raises(ValueError):
        estimator.setLearningRate(-1)

    with pytest.raises(ValueError):
        estimator.setLearningRate("some-value")

    estimator.setLearningRate("0.1")
    estimator._transfer_params_to_java()
    assert estimator.getLearningRate() == estimator._call_java("getLearningRate")


def test_linearLearnerBinaryClassifier_validates_beta_1():

    estimator = get_linear_learner_binary_classifier()
    estimator.setBeta1(0.5)
    assert estimator.getBeta1() == 0.5

    with pytest.raises(ValueError):
        estimator.setBeta1(3)

    estimator.setBeta1(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getBeta1() == estimator._call_java("getBeta1")


def test_linearLearnerBinaryClassifier_validates_beta_2():

    estimator = get_linear_learner_binary_classifier()
    estimator.setBeta2(0.5)
    assert estimator.getBeta2() == 0.5

    with pytest.raises(ValueError):
        estimator.setBeta2(3)

    estimator.setBeta2(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getBeta2() == estimator._call_java("getBeta2")


def test_linearLearnerBinaryClassifier_validates_bias_lr_mult():

    estimator = get_linear_learner_binary_classifier()
    estimator.setBiasLrMult(0.5)
    assert estimator.getBiasLrMult() == 0.5

    with pytest.raises(ValueError):
        estimator.setBiasLrMult(0)

    estimator.setBiasLrMult(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getBiasLrMult() == estimator._call_java("getBiasLrMult")


def test_linearLearnerBinaryClassifier_validates_bias_wd_mult():

    estimator = get_linear_learner_binary_classifier()
    estimator.setBiasWdMult(0.5)
    assert estimator.getBiasWdMult() == 0.5

    with pytest.raises(ValueError):
        estimator.setBiasWdMult(-1)

    estimator.setBiasWdMult(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getBiasWdMult() == estimator._call_java("getBiasWdMult")


def test_linearLearnerBinaryClassifier_validates_use_lr_scheduler():

    estimator = get_linear_learner_binary_classifier()
    estimator.setUseLrScheduler("True")
    assert estimator.getUseLrScheduler() is True

    with pytest.raises(ValueError):
        estimator.setUseLrScheduler("some-value")

    estimator.setUseLrScheduler("False")
    estimator._transfer_params_to_java()
    assert estimator.getUseLrScheduler() == estimator._call_java("getUseLrScheduler")


def test_linearLearnerBinaryClassifier_validates_lr_scheduler_step():

    estimator = get_linear_learner_binary_classifier()
    estimator.setLrSchedulerStep(5)
    assert estimator.getLrSchedulerStep() == 5

    with pytest.raises(ValueError):
        estimator.setLrSchedulerStep(0)

    estimator.setLrSchedulerStep(1)
    estimator._transfer_params_to_java()
    assert estimator.getLrSchedulerStep() == estimator._call_java("getLrSchedulerStep")


def test_linearLearnerBinaryClassifier_validates_lr_scheduler_factor():

    estimator = get_linear_learner_binary_classifier()
    estimator.setLrSchedulerFactor(0.5)
    assert estimator.getLrSchedulerFactor() == 0.5

    with pytest.raises(ValueError):
        estimator.setLrSchedulerFactor(3)

    estimator.setLrSchedulerFactor(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getLrSchedulerFactor() == estimator._call_java("getLrSchedulerFactor")


def test_linearLearnerBinaryClassifier_validates_lr_scheduler_minimum_lr():

    estimator = get_linear_learner_binary_classifier()
    estimator.setLrSchedulerMinimumLr(0.5)
    assert estimator.getLrSchedulerMinimumLr() == 0.5

    with pytest.raises(ValueError):
        estimator.setLrSchedulerMinimumLr(0)

    estimator.setLrSchedulerMinimumLr(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getLrSchedulerMinimumLr() == estimator._call_java("getLrSchedulerMinimumLr")


def test_linearLearnerBinaryClassifier_validates_binary_classifier_model_selection_criteria():

    estimator = get_linear_learner_binary_classifier()
    estimator.setBinaryClassifierModelSelectionCriteria("accuracy")
    assert estimator.getBinaryClassifierModelSelectionCriteria() == "accuracy"

    with pytest.raises(ValueError):
        estimator.setBinaryClassifierModelSelectionCriteria("some-value")

    estimator.setBinaryClassifierModelSelectionCriteria("f1")
    estimator._transfer_params_to_java()
    assert estimator.getBinaryClassifierModelSelectionCriteria() \
        == estimator._call_java("getBinaryClassifierModelSelectionCriteria")


def test_linearLearnerBinaryClassifier_validates_target_recall():

    estimator = get_linear_learner_binary_classifier()
    estimator.setTargetRecall(0.5)
    assert estimator.getTargetRecall() == 0.5

    with pytest.raises(ValueError):
        estimator.setTargetRecall(3)

    estimator.setTargetRecall(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getTargetRecall() == estimator._call_java("getTargetRecall")


def test_linearLearnerBinaryClassifier_validates_target_precision():

    estimator = get_linear_learner_binary_classifier()
    estimator.setTargetPrecision(0.5)
    assert estimator.getTargetPrecision() == 0.5

    with pytest.raises(ValueError):
        estimator.setTargetPrecision(3)

    estimator.setTargetPrecision(0.1)
    estimator._transfer_params_to_java()
    assert estimator.getTargetPrecision() == estimator._call_java("getTargetPrecision")


def test_linearLearnerRegressor_validates_normalize_data():

    estimator = get_linear_learner_binary_classifier()
    estimator.setNormalizeData("True")
    assert estimator.getNormalizeData() is True

    with pytest.raises(ValueError):
        estimator.setNormalizeData("some-value")

    estimator.setNormalizeData("False")
    estimator._transfer_params_to_java()
    assert estimator.getNormalizeData() == estimator._call_java("getNormalizeData")


def test_linearLearnerRegressor_validates_normalize_label():

    estimator = get_linear_learner_binary_classifier()
    estimator.setNormalizeLabel("True")
    assert estimator.getNormalizeLabel() is True

    with pytest.raises(ValueError):
        estimator.setNormalizeLabel("some-value")

    estimator.setNormalizeLabel("False")
    estimator._transfer_params_to_java()
    assert estimator.getNormalizeLabel() == estimator._call_java("getNormalizeLabel")


def test_linearLearnerRegressor_validates_unbias_data():

    estimator = get_linear_learner_binary_classifier()
    estimator.setUnbiasData("True")
    assert estimator.getUnbiasData() is True

    with pytest.raises(ValueError):
        estimator.setUnbiasData("some-value")

    estimator.setUnbiasData("False")
    estimator._transfer_params_to_java()
    assert estimator.getUnbiasData() == estimator._call_java("getUnbiasData")


def test_linearLearnerRegressor_validates_unbias_label():

    estimator = get_linear_learner_binary_classifier()
    estimator.setUnbiasLabel("True")
    assert estimator.getUnbiasLabel() is True

    with pytest.raises(ValueError):
        estimator.setUnbiasLabel("some-value")

    estimator.setUnbiasLabel("False")
    estimator._transfer_params_to_java()
    assert estimator.getUnbiasLabel() == estimator._call_java("getUnbiasLabel")


def test_linearLearnerBinaryClassifier_num_point_for_scaler():

    estimator = get_linear_learner_binary_classifier()
    estimator.setNumPointForScaler(5)
    assert estimator.getNumPointForScaler() == 5

    with pytest.raises(ValueError):
        estimator.setNumPointForScaler(-1)

    estimator.setNumPointForScaler(1)
    estimator._transfer_params_to_java()
    assert estimator.getNumPointForScaler() == estimator._call_java("getNumPointForScaler")
