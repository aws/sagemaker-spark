/*
 * Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *   http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazonaws.services.sagemaker.sparksdk.algorithms

import scala.collection.JavaConverters.mapAsJavaMapConverter

import com.amazonaws.regions.Regions
import org.scalatest.FlatSpec
import org.scalatest.mockito.MockitoSugar

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.{LinearLearnerBinaryClassifierProtobufResponseRowDeserializer, LinearLearnerRegressorProtobufResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

class LinearLearnerSageMakerEstimatorTests extends FlatSpec with MockitoSugar {

  def createLinearLearnerRegressor(region: String =
                                   Regions.US_WEST_2.getName): LinearLearnerSageMakerEstimator = {
    new LinearLearnerRegressor(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  def createLinearLearnerBinaryClassifier(region: String =
                                   Regions.US_WEST_2.getName) : LinearLearnerBinaryClassifier = {
    new LinearLearnerBinaryClassifier(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  it should "use the correct defaults for binary classifier" in {
    val estimator = createLinearLearnerBinaryClassifier()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.
      isInstanceOf[LinearLearnerBinaryClassifierProtobufResponseRowDeserializer])
  }

  it should "use the correct images in all regions for binary classifier" in {
    val estimatorUSEast1 = createLinearLearnerBinaryClassifier(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:1")

    val estimatorUSEast2 = createLinearLearnerBinaryClassifier(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:1")

    val estimatorEUWest1 = createLinearLearnerBinaryClassifier(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage ==
      "438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:1")

    val estimatorUSWest2 = createLinearLearnerBinaryClassifier(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage ==
      "174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1")
  }

  it should "use the correct defaults for regressor" in {
    val estimator = createLinearLearnerRegressor()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.
      isInstanceOf[LinearLearnerRegressorProtobufResponseRowDeserializer])
  }

  it should "use the correct images in all regions for regressor" in {
    val estimatorUSEast1 = createLinearLearnerRegressor(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:1")

    val estimatorUSEast2 = createLinearLearnerRegressor(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:1")

    val estimatorEUWest1 = createLinearLearnerRegressor(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage ==
      "438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:1")

    val estimatorUSWest2 = createLinearLearnerRegressor(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage ==
      "174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1")
  }

  it should "setFeatureDim" in {
    val estimator = createLinearLearnerRegressor()
    val featureDim = 100
    estimator.setFeatureDim(featureDim)
    assert(featureDim == estimator.getFeatureDim)
  }

  it should "setMiniBatchSize" in {
    val estimator = createLinearLearnerRegressor()
    val miniBatchSize = 100
    estimator.setMiniBatchSize(miniBatchSize)
    assert(miniBatchSize == estimator.getMiniBatchSize)
  }

  it should "setEpochs" in {
    val estimator = createLinearLearnerRegressor()
    val epochs = 10
    estimator.setEpochs(epochs)
    assert(epochs == estimator.getEpochs)
  }

  it should "setUseBias" in {
    val estimator = createLinearLearnerRegressor()
    val useBias = true
    estimator.setUseBias(useBias)
    assert(useBias == estimator.getUseBias)
  }

  it should "setBinaryClassifierModelSelectionCriteria" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val binaryClassifierModelSelectionCriteria = "accuracy"
    estimator.setBinaryClassifierModelSelectionCriteria(binaryClassifierModelSelectionCriteria)
    assert(binaryClassifierModelSelectionCriteria ==
      estimator.getBinaryClassifierModelSelectionCriteria)
  }

  it should "setTargetRecall" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val targetRecall = 0.8
    estimator.setTargetRecall(targetRecall)
    assert(targetRecall == estimator.getTargetRecall)
  }

  it should "setTargetPrecision" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val targetPrecision = 0.8
    estimator.setTargetPrecision(targetPrecision)
    assert(targetPrecision == estimator.getTargetPrecision)
  }

  it should "setPositiveExampleWeightMult with a double" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val positiveExampleWeightMult = 0.5
    estimator.setPositiveExampleWeightMult(positiveExampleWeightMult)
    assert(positiveExampleWeightMult.toString == estimator.getPositiveExampleWeightMult)
  }

  it should "setPositiveExampleWeightMult with a string" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val positiveExampleWeightMult = "balanced"
    estimator.setPositiveExampleWeightMult(positiveExampleWeightMult)
    assert(positiveExampleWeightMult == estimator.getPositiveExampleWeightMult)
  }

  it should "setNumModels with an int" in {
    val estimator = createLinearLearnerRegressor()
    val numModels = 5
    estimator.setNumModels(numModels)
    assert(numModels.toString == estimator.getNumModels)
  }

  it should "setNumModels with a string" in {
    val estimator = createLinearLearnerRegressor()
    val numModels = "5"
    estimator.setNumModels(numModels)
    assert(numModels == estimator.getNumModels)
  }

  it should "setNumCalibrationSamples" in {
    val estimator = createLinearLearnerRegressor()
    val numCalibrationSamples = 16
    estimator.setNumCalibrationSamples(numCalibrationSamples)
    assert(numCalibrationSamples == estimator.getNumCalibrationSamples)
  }

  it should "setInitMethod" in {
    val estimator = createLinearLearnerRegressor()
    val initMethod = "uniform"
    estimator.setInitMethod(initMethod)
    assert(initMethod == estimator.getInitMethod)
  }

  it should "setInitScale" in {
    val estimator = createLinearLearnerRegressor()
    val initScale = 0.5
    estimator.setInitScale(initScale)
    assert(initScale == estimator.getInitScale)
  }

  it should "setInitSigma" in {
    val estimator = createLinearLearnerRegressor()
    val initSigma = 0.01
    estimator.setInitSigma(initSigma)
    assert(initSigma == estimator.getInitSigma)
  }

  it should "setInitBias" in {
    val estimator = createLinearLearnerRegressor()
    val initBias = 100
    estimator.setInitBias(initBias)
    assert(initBias == estimator.getInitBias)
  }

  it should "setOptimizer" in {
    val estimator = createLinearLearnerRegressor()
    val optimizer = "adam"
    estimator.setOptimizer(optimizer)
    assert(optimizer == estimator.getOptimizer)
  }

  it should "setLoss" in {
    val estimator = createLinearLearnerRegressor()
    val loss = "logistic"
    estimator.setLoss(loss)
    assert(loss == estimator.getLoss)
  }

  it should "setWd" in {
    val estimator = createLinearLearnerRegressor()
    val wd = 0.5
    estimator.setWd(wd)
    assert(wd == estimator.getWd)
  }

  it should "setL1" in {
    val estimator = createLinearLearnerRegressor()
    val l1 = 0.0
    estimator.setL1(l1)
    assert(l1 == estimator.getL1)
  }

  it should "setMomentum" in {
    val estimator = createLinearLearnerRegressor()
    val momentum = 0.9999
    estimator.setMomentum(momentum)
    assert(momentum == estimator.getMomentum)
  }

  it should "setLearningRate with a double" in {
    val estimator = createLinearLearnerRegressor()
    val learningRate = 0.1
    estimator.setLearningRate(learningRate)
    assert(learningRate.toString == estimator.getLearningRate)
  }

  it should "setLearningRate with a string" in {
    val estimator = createLinearLearnerRegressor()
    val learningRate = "auto"
    estimator.setLearningRate(learningRate)
    assert(learningRate == estimator.getLearningRate)
  }

  it should "setBeta1" in {
    val estimator = createLinearLearnerRegressor()
    val beta1 = 0.9
    estimator.setBeta1(beta1)
    assert(beta1 == estimator.getBeta1)
  }

  it should "setBeta2" in {
    val estimator = createLinearLearnerRegressor()
    val beta2 = 0.999
    estimator.setBeta2(beta2)
    assert(beta2 == estimator.getBeta2)
  }

  it should "setBiasLrMult" in {
    val estimator = createLinearLearnerRegressor()
    val biasLrMult = 0.1
    estimator.setBiasLrMult(biasLrMult)
    assert(biasLrMult == estimator.getBiasLrMult)
  }

  it should "setBiasWdMult" in {
    val estimator = createLinearLearnerRegressor()
    val biasWdMult = 0.1
    estimator.setBiasWdMult(biasWdMult)
    assert(biasWdMult == estimator.getBiasWdMult)
  }

  it should "setUseLrScheduler" in {
    val estimator = createLinearLearnerRegressor()
    val useLrScheduler = true
    estimator.setUseLrScheduler(useLrScheduler)
    assert(useLrScheduler == estimator.getUseLrScheduler)
  }

  it should "setLrSchedulerStep" in {
    val estimator = createLinearLearnerRegressor()
    val lrSchedulerStep = 1
    estimator.setLrSchedulerStep(lrSchedulerStep)
    assert(lrSchedulerStep == estimator.getLrSchedulerStep)
  }

  it should "setLrSchedulerFactor" in {
    val estimator = createLinearLearnerRegressor()
    val lrSchedulerFactor = 0.1
    estimator.setLrSchedulerFactor(lrSchedulerFactor)
    assert(lrSchedulerFactor == estimator.getLrSchedulerFactor)
  }

  it should "setLrSchedulerMinimumLr" in {
    val estimator = createLinearLearnerRegressor()
    val lrSchedulerMinimumLr = 0.1
    estimator.setLrSchedulerMinimumLr(lrSchedulerMinimumLr)
    assert(lrSchedulerMinimumLr == estimator.getLrSchedulerMinimumLr)
  }

  it should "setNormalizeData" in {
    val estimator = createLinearLearnerRegressor()
    val normalizeData = true
    estimator.setNormalizeData(normalizeData)
    assert(normalizeData == estimator.getNormalizeData)
  }

  it should "setNormalizeLabel" in {
    val estimator = createLinearLearnerRegressor()
    val normalizeLabel = true
    estimator.setNormalizeLabel(normalizeLabel)
    assert(normalizeLabel == estimator.getNormalizeLabel)
  }

  it should "setUnbiasData" in {
    val estimator = createLinearLearnerRegressor()
    val unbiasData = true
    estimator.setUnbiasData(unbiasData)
    assert(unbiasData == estimator.getUnbiasData)
  }

  it should "setUnbiasLabel" in {
    val estimator = createLinearLearnerRegressor()
    val unbiasLabel = true
    estimator.setUnbiasLabel(unbiasLabel)
    assert(unbiasLabel == estimator.getUnbiasLabel)
  }

  it should "setNumPointForScaler" in {
    val estimator = createLinearLearnerRegressor()
    val numPointForScaler = 5
    estimator.setNumPointForScaler(numPointForScaler)
    assert(numPointForScaler == estimator.getNumPointForScaler)
  }

  it should "setEarlyStoppingPatience" in {
    val estimator = createLinearLearnerRegressor()
    val earlyStoppingPatience = 5
    estimator.setEarlyStoppingPatience(earlyStoppingPatience)
    assert(earlyStoppingPatience == estimator.getEarlyStoppingPatience)
  }

  it should "setEarlyStoppingTolerance" in {
    val estimator = createLinearLearnerRegressor()
    val earlyStoppingTolerance = 0.5
    estimator.setEarlyStoppingTolerance(earlyStoppingTolerance)
    assert(earlyStoppingTolerance == estimator.getEarlyStoppingTolerance)
  }

  it should "setMargin" in {
    val estimator = createLinearLearnerRegressor()
    val margin = 5.0
    estimator.setMargin(margin)
    assert(margin == estimator.getMargin)
  }

  it should "setQuantile" in {
    val estimator = createLinearLearnerRegressor()
    val quantile = 0.5
    estimator.setQuantile(quantile)
    assert(quantile == estimator.getQuantile)
  }

  it should "setLossInsensitivity" in {
    val estimator = createLinearLearnerRegressor()
    val lossInsensitivity = 0.5
    estimator.setLossInsensitivity(lossInsensitivity)
    assert(lossInsensitivity == estimator.getLossInsensitivity)
  }

  it should "setHuberDelta" in {
    val estimator = createLinearLearnerRegressor()
    val huberDelta = 5.0
    estimator.setHuberDelta(huberDelta)
    assert(huberDelta == estimator.getHuberDelta)
  }

  it should "create correct hyper-parameter map" in {
    val estimator = createLinearLearnerBinaryClassifier()
    estimator.setEpochs(2)
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    estimator.setUseBias(false)
    estimator.setBinaryClassifierModelSelectionCriteria("f1")
    estimator.setTargetRecall(0.8)
    estimator.setTargetPrecision(0.8)
    estimator.setPositiveExampleWeightMult("balanced")
    estimator.setNumModels("auto")
    estimator.setNumCalibrationSamples(5)
    estimator.setInitMethod("uniform")
    estimator.setInitScale(0.07)
    estimator.setInitSigma(0.01)
    estimator.setInitBias(0)
    estimator.setOptimizer("sgd")
    estimator.setLoss("squared_loss")
    estimator.setWd(0.5)
    estimator.setL1(0.5)
    estimator.setMomentum(0)
    estimator.setLearningRate(0.5)
    estimator.setBeta1(0.9)
    estimator.setBeta2(0.999)
    estimator.setBiasLrMult(0.5)
    estimator.setBiasWdMult(0.5)
    estimator.setUseLrScheduler(false)
    estimator.setLrSchedulerStep(5)
    estimator.setLrSchedulerFactor(0.5)
    estimator.setLrSchedulerMinimumLr(0.5)
    estimator.setNormalizeData(false)
    estimator.setNormalizeLabel(false)
    estimator.setUnbiasData(false)
    estimator.setUnbiasLabel(false)
    estimator.setNumPointForScaler(100)
    estimator.setEarlyStoppingPatience(5)
    estimator.setEarlyStoppingTolerance(0.1)
    estimator.setMargin(0.5)
    estimator.setQuantile(0.5)
    estimator.setLossInsensitivity(0.05)
    estimator.setHuberDelta(0.5)

    val hyperParamMap = Map(
      "predictor_type" -> "binary_classifier",
      "epochs" -> "2",
      "feature_dim" -> "2",
      "mini_batch_size" -> "2",
      "use_bias" -> "False",
      "binary_classifier_model_selection_criteria" -> "f1",
      "target_recall" -> "0.8",
      "target_precision" -> "0.8",
      "positive_example_weight_mult" -> "balanced",
      "num_models" -> "auto",
      "num_calibration_samples" -> "5",
      "init_method" -> "uniform",
      "init_scale" -> "0.07",
      "init_sigma" -> "0.01",
      "init_bias" -> "0.0",
      "optimizer" -> "sgd",
      "loss" -> "squared_loss",
      "wd" -> "0.5",
      "l1" -> "0.5",
      "momentum" -> "0.0",
      "learning_rate" -> "0.5",
      "beta_1" -> "0.9",
      "beta_2" -> "0.999",
      "bias_lr_mult" -> "0.5",
      "bias_wd_mult" -> "0.5",
      "use_lr_scheduler" -> "False",
      "lr_scheduler_step" -> "5",
      "lr_scheduler_factor" -> "0.5",
      "lr_scheduler_minimum_lr" -> "0.5",
      "normalize_data" -> "False",
      "normalize_label" -> "False",
      "unbias_data" -> "False",
      "unbias_label" -> "False",
      "num_point_for_scaler" -> "100",
      "early_stopping_patience" -> "5",
      "early_stopping_tolerance" -> "0.1",
      "margin" -> "0.5",
      "quantile" -> "0.5",
      "loss_insensitivity" -> "0.05",
      "huber_delta" -> "0.5"
    )

    assert(hyperParamMap.asJava == estimator.makeHyperParameters())
  }

  it should "validate featureDim" in {
    val estimator = createLinearLearnerRegressor()
    val caughtFeatureDim = intercept[IllegalArgumentException] {
      estimator.setFeatureDim(-1)
    }
  }

  it should "validate miniBatchSize" in {
    val estimator = createLinearLearnerRegressor()
    val caughtMiniBatchSize = intercept[IllegalArgumentException] {
      estimator.setMiniBatchSize(-1)
    }
  }

  it should "validate epochs" in {
    val estimator = createLinearLearnerRegressor()
    val caughtEpochs = intercept[IllegalArgumentException] {
      estimator.setEpochs(-1)
    }
  }

  it should "validate binaryClassifierModelSelectionCriteria" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val caughtBinaryClassifierModelSelectionCriteria = intercept[IllegalArgumentException] {
      estimator.setBinaryClassifierModelSelectionCriteria("invalid")
    }
  }

  it should "validate targetRecall" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val caughtTargetRecall = intercept[IllegalArgumentException] {
      estimator.setTargetRecall(-1.0)
    }
  }

  it should "validate targetPrecision" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val caughtTargetPrecision = intercept[IllegalArgumentException] {
      estimator.setTargetPrecision(-1.0)
    }
  }

  it should "validate positiveExampleWeightMult with a double" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val positiveExampleWeightMult = intercept[IllegalArgumentException] {
      estimator.setPositiveExampleWeightMult(-1.0)
    }
  }

  it should "validate positiveExampleWeightMult with a string" in {
    val estimator = createLinearLearnerBinaryClassifier()
    val positiveExampleWeightMult = intercept[IllegalArgumentException] {
      estimator.setPositiveExampleWeightMult("invalid")
    }
  }

  it should "validate numModels when set with an int" in {
    val estimator = createLinearLearnerRegressor()
    val caughtNumModels = intercept[IllegalArgumentException] {
      estimator.setNumModels(-1)
    }
  }

  it should "validate numModels when set with a string" in {
    val estimator = createLinearLearnerRegressor()
    val caughtNumModels = intercept[IllegalArgumentException] {
      estimator.setNumModels("invalid")
    }
  }

  it should "validate numCalibrationSamples when set with an int" in {
    val estimator = createLinearLearnerRegressor()
    val caughtNumCalibrationSamples = intercept[IllegalArgumentException] {
      estimator.setNumCalibrationSamples(-1)
    }
  }

  it should "validate initMethod" in {
    val estimator = createLinearLearnerRegressor()
    val caughtInitMethod = intercept[IllegalArgumentException] {
      estimator.setInitMethod("invalid")
    }
  }

  it should "validate initScale" in {
    val estimator = createLinearLearnerRegressor()
    val caughtInitScale = intercept[IllegalArgumentException] {
      estimator.setInitScale(-2.0)
    }
  }

  it should "validate initSigma" in {
    val estimator = createLinearLearnerRegressor()
    val caughtInitSigma = intercept[IllegalArgumentException] {
      estimator.setInitSigma(-1.0)
    }
  }

  it should "validate initBias" in {
    val estimator = createLinearLearnerRegressor()
    estimator.setInitBias(-9999)
  }

  it should "validate optimizer" in {
    val estimator = createLinearLearnerRegressor()
    val caughtOptimizer = intercept[IllegalArgumentException] {
      estimator.setOptimizer("invalid")
    }
  }

  it should "validate loss" in {
    val estimator = createLinearLearnerRegressor()
    val caughtLoss = intercept[IllegalArgumentException] {
      estimator.setLoss("invalid")
    }
  }

  it should "validate wd when set with a double" in {
    val estimator = createLinearLearnerRegressor()
    val caughtWd = intercept[IllegalArgumentException] {
      estimator.setWd(-1.0)
    }
  }

  it should "validate l1 when set with a double" in {
    val estimator = createLinearLearnerRegressor()
    val caughtL1 = intercept[IllegalArgumentException] {
      estimator.setL1(-1.0)
    }
  }

  it should "validate momentum" in {
    val estimator = createLinearLearnerRegressor()
    val caughtMomentum = intercept[IllegalArgumentException] {
      estimator.setMomentum(-1.0)
    }
  }

  it should "validate learningRate when set with a double" in {
    val estimator = createLinearLearnerRegressor()
    val caughtLearningRate = intercept[IllegalArgumentException] {
      estimator.setLearningRate(-1.0)
    }
  }

  it should "validate learningRate when set with a string" in {
    val estimator = createLinearLearnerRegressor()
    val caughtLearningRate = intercept[IllegalArgumentException] {
      estimator.setLearningRate("invalid")
    }
  }

  it should "validate beta1" in {
    val estimator = createLinearLearnerRegressor()
    val caughtBeta1 = intercept[IllegalArgumentException] {
      estimator.setBeta1(-1.0)
    }
  }

  it should "validate beta2" in {
    val estimator = createLinearLearnerRegressor()
    val caughtBeta2 = intercept[IllegalArgumentException] {
      estimator.setBeta2(-1.0)
    }
  }

  it should "validate biasLrMult" in {
    val estimator = createLinearLearnerRegressor()
    val caughtBiasLrMult = intercept[IllegalArgumentException] {
      estimator.setBiasLrMult(-1.0)
    }
  }

  it should "validate biasWdMult" in {
    val estimator = createLinearLearnerRegressor()
    val caughtBiasWdMult = intercept[IllegalArgumentException] {
      estimator.setBiasWdMult(-1.0)
    }
  }

  it should "validate lrSchedulerStep" in {
    val estimator = createLinearLearnerRegressor()
    val caughtLrSchedulerStep = intercept[IllegalArgumentException] {
      estimator.setLrSchedulerStep(-1)
    }
  }

  it should "validate lrSchedulerFactor" in {
    val estimator = createLinearLearnerRegressor()
    val caughtLrSchedulerFactor = intercept[IllegalArgumentException] {
      estimator.setLrSchedulerFactor(-1.0)
    }
  }

  it should "validate lrSchedulerMinimumLr" in {
    val estimator = createLinearLearnerRegressor()
    val caughtLrSchedulerMinimumLr = intercept[IllegalArgumentException] {
      estimator.setLrSchedulerMinimumLr(-1.0)
    }
  }

  it should "validate numPointForScaler" in {
    val estimator = createLinearLearnerRegressor()
    val caughtNumPointForScaler = intercept[IllegalArgumentException] {
      estimator.setNumPointForScaler(-1)
    }
  }

  it should "validate earlyStoppingPatience" in {
    val estimator = createLinearLearnerRegressor()
    val earlyStoppingPatience = intercept[IllegalArgumentException] {
      estimator.setEarlyStoppingPatience(-1)
    }
  }

  it should "validate earlyStoppingTolerance" in {
    val estimator = createLinearLearnerRegressor()
    val earlyStoppingTolerance = intercept[IllegalArgumentException] {
      estimator.setEarlyStoppingTolerance(-1.0)
    }
  }

  it should "validate margin" in {
    val estimator = createLinearLearnerRegressor()
    val margin = intercept[IllegalArgumentException] {
      estimator.setMargin(-1.0)
    }
  }

  it should "validate quantile" in {
    val estimator = createLinearLearnerRegressor()
    val quantile = intercept[IllegalArgumentException] {
      estimator.setQuantile(2.0)
    }
  }

  it should "validate lossInsensitivity" in {
    val estimator = createLinearLearnerRegressor()
    val lossInsensitivity = intercept[IllegalArgumentException] {
      estimator.setLossInsensitivity(-1.0)
    }
  }

  it should "validate huberDelta" in {
    val estimator = createLinearLearnerRegressor()
    val huberDelta = intercept[IllegalArgumentException] {
      estimator.setHuberDelta(-1.0)
    }
  }
}
