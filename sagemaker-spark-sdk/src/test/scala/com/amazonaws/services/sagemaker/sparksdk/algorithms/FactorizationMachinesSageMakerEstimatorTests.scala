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

import java.util.NoSuchElementException

import scala.collection.JavaConverters.mapAsJavaMapConverter

import com.amazonaws.regions.Regions
import org.scalatest.FlatSpec
import org.scalatest.mockito.MockitoSugar

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.{FactorizationMachinesBinaryClassifierDeserializer, FactorizationMachinesRegressorDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

class FactorizationMachinesSageMakerEstimatorTests extends FlatSpec with MockitoSugar {
  def createFactorizationMachinesRegressor(region: String = Regions.US_WEST_2.getName):
    FactorizationMachinesSageMakerEstimator = {
    new FactorizationMachinesRegressor(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  def createFactorizationMachinesBinaryClassifier(region: String = Regions.US_WEST_2.getName):
    FactorizationMachinesSageMakerEstimator = {
    new FactorizationMachinesBinaryClassifier(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  it should "use the correct defaults for binary classifier" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.
      isInstanceOf[FactorizationMachinesBinaryClassifierDeserializer])
  }

  it should "use the correct images in all regions for binary classifier" in {
    val estimatorUSEast1 =
      createFactorizationMachinesBinaryClassifier(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "382416733822.dkr.ecr.us-east-1.amazonaws.com/factorization-machines:1")

    val estimatorUSEast2 =
      createFactorizationMachinesBinaryClassifier(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "404615174143.dkr.ecr.us-east-2.amazonaws.com/factorization-machines:1")

    val estimatorEUWest1 =
      createFactorizationMachinesBinaryClassifier(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage ==
      "438346466558.dkr.ecr.eu-west-1.amazonaws.com/factorization-machines:1")

    val estimatorUSWest2 =
      createFactorizationMachinesBinaryClassifier(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage ==
      "174872318107.dkr.ecr.us-west-2.amazonaws.com/factorization-machines:1")

    val estimatorAPNorthEast1 =
      createFactorizationMachinesBinaryClassifier(region = Regions.AP_NORTHEAST_1.getName)
    assert(estimatorAPNorthEast1.trainingImage ==
      "351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/factorization-machines:1")

    val estimatorAPNorthEast2 =
      createFactorizationMachinesBinaryClassifier(region = Regions.AP_NORTHEAST_2.getName)
    assert(estimatorAPNorthEast2.trainingImage ==
      "835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/factorization-machines:1")
  }

  it should "use the correct defaults for regressor" in {
    val estimator = createFactorizationMachinesRegressor()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.
      isInstanceOf[FactorizationMachinesRegressorDeserializer])
  }

  it should "use the correct images in all regions for regressor" in {
    val estimatorUSEast1 = createFactorizationMachinesRegressor(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "382416733822.dkr.ecr.us-east-1.amazonaws.com/factorization-machines:1")

    val estimatorUSEast2 = createFactorizationMachinesRegressor(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "404615174143.dkr.ecr.us-east-2.amazonaws.com/factorization-machines:1")

    val estimatorEUWest1 = createFactorizationMachinesRegressor(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage ==
      "438346466558.dkr.ecr.eu-west-1.amazonaws.com/factorization-machines:1")

    val estimatorUSWest2 = createFactorizationMachinesRegressor(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage ==
      "174872318107.dkr.ecr.us-west-2.amazonaws.com/factorization-machines:1")

    val estimatorAPNorthEast1 =
      createFactorizationMachinesRegressor(region = Regions.AP_NORTHEAST_1.getName)
    assert(estimatorAPNorthEast1.trainingImage ==
      "351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/factorization-machines:1")

    val estimatorAPNorthEast2 =
      createFactorizationMachinesRegressor(region = Regions.AP_NORTHEAST_2.getName)
    assert(estimatorAPNorthEast2.trainingImage ==
      "835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/factorization-machines:1")
  }

  it should "setFeatureDim" in {
    val estimator = createFactorizationMachinesRegressor()
    val featureDim = 100
    estimator.setFeatureDim(featureDim)
    assert(featureDim == estimator.getFeatureDim)
  }

  it should "setMiniBatchSize" in {
    val estimator = createFactorizationMachinesRegressor()
    val miniBatchSize = 100
    estimator.setMiniBatchSize(miniBatchSize)
    assert(miniBatchSize == estimator.getMiniBatchSize)
  }

  it should "setNumFactors" in {
    val estimator = createFactorizationMachinesRegressor()
    val numFactors = 10
    estimator.setNumFactors(numFactors)
    assert(numFactors == estimator.getNumFactors)
  }

  it should "setEpochs" in {
    val estimator = createFactorizationMachinesRegressor()
    val epochs = 10
    estimator.setEpochs(epochs)
    assert(epochs == estimator.getEpochs)
  }

  it should "setClipGradient" in {
    val estimator = createFactorizationMachinesRegressor()
    val clipGradient = 1.1
    estimator.setClipGradient(clipGradient)
    assert(clipGradient == estimator.getClipGradient)
  }

  it should "setEps" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    val eps = 0.01
    estimator.setEps(eps)
    assert(eps == estimator.getEps)
  }

  it should "setRescaleGrad" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    val rescaleGrad = 0.08
    estimator.setRescaleGrad(rescaleGrad)
    assert(rescaleGrad == estimator.getRescaleGrad)
  }

  it should "setBiasLr" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    val biasLr = 3.8
    estimator.setBiasLr(biasLr)
    assert(biasLr == estimator.getBiasLr)
  }

  it should "setLinearLr" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearLr = 5
    estimator.setLinearLr(linearLr)
    assert(linearLr == estimator.getLinearLr)
  }

  it should "setFactorsLr" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsLr = 3
    estimator.setFactorsLr(factorsLr)
    assert(factorsLr == estimator.getFactorsLr)
  }

  it should "setBiasWd" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasWd = 1.6
    estimator.setBiasWd(biasWd)
    assert(biasWd == estimator.getBiasWd)
  }

  it should "setLinearWd" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearWd = 2.2
    estimator.setLinearWd(linearWd)
    assert(linearWd == estimator.getLinearWd)
  }

  it should "setFactorsWd" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsWd = 0.5
    estimator.setFactorsWd(factorsWd)
    assert(factorsWd == estimator.getFactorsWd)
  }

  it should "setBiasInitMethod" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitMethod = "normal"
    estimator.setBiasInitMethod(biasInitMethod)
    assert(biasInitMethod == estimator.getBiasInitMethod)
  }

  it should "setBiasInitScale" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitScale = 10.10
    estimator.setBiasInitScale(biasInitScale)
    assert(biasInitScale == estimator.getBiasInitScale)
  }

  it should "setBiasInitSigma" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitSigma = 1.11
    estimator.setBiasInitSigma(biasInitSigma)
    assert(biasInitSigma == estimator.getBiasInitSigma)
  }

  it should "setBiasInitValue" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitValue = 11.1
    estimator.setBiasInitValue(biasInitValue)
    assert(biasInitValue == estimator.getBiasInitValue)
  }

  it should "setLinearInitMethod" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitMethod = "normal"
    estimator.setLinearInitMethod(linearInitMethod)
    assert(linearInitMethod == estimator.getLinearInitMethod)
  }

  it should "setLinearInitScale" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitScale = 10.10
    estimator.setLinearInitScale(linearInitScale)
    assert(linearInitScale == estimator.getLinearInitScale)
  }

  it should "setLinearInitSigma" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitSigma = 1.11
    estimator.setLinearInitSigma(linearInitSigma)
    assert(linearInitSigma == estimator.getLinearInitSigma)
  }

  it should "setLinearInitValue" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitValue = 11.1
    estimator.setLinearInitValue(linearInitValue)
    assert(linearInitValue == estimator.getLinearInitValue)
  }

  it should "setFactorsInitMethod" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitMethod = "uniform"
    estimator.setFactorsInitMethod(factorsInitMethod)
    assert(factorsInitMethod == estimator.getFactorsInitMethod)
  }

  it should "setFactorsInitScale" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitScale = 10.10
    estimator.setFactorsInitScale(factorsInitScale)
    assert(factorsInitScale == estimator.getFactorsInitScale)
  }

  it should "setFactorsInitSigma" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitSigma = 1.11
    estimator.setFactorsInitSigma(factorsInitSigma)
    assert(factorsInitSigma == estimator.getFactorsInitSigma)
  }

  it should "setFactorsInitValue" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitValue = 11.1
    estimator.setFactorsInitValue(factorsInitValue)
    assert(factorsInitValue == estimator.getFactorsInitValue)
  }

  it should "validate setFeatureDim" in {
    val estimator = createFactorizationMachinesRegressor()
    val featureDim = -1
    val caught = intercept[IllegalArgumentException] {
      estimator.setFeatureDim(featureDim)
    }
  }

  it should "validate setMiniBatchSize" in {
    val estimator = createFactorizationMachinesRegressor()
    val miniBatchSize = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setMiniBatchSize(miniBatchSize)
    }
  }

  it should "validate setNumFactors" in {
    val estimator = createFactorizationMachinesRegressor()
    val numFactors = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setNumFactors(numFactors)
    }
  }

  it should "validate setEpochs" in {
    val estimator = createFactorizationMachinesRegressor()
    val epochs = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setEpochs(epochs)
    }
  }

  it should "validate setBiasLr" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    val biasLr = -3.8
    val caught = intercept[IllegalArgumentException] {
      estimator.setBiasLr(biasLr)
    }
  }

  it should "validate setLinearLr" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearLr = -5
    val caught = intercept[IllegalArgumentException] {
      estimator.setLinearLr(linearLr)
    }
  }

  it should "validate setFactorsLr" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsLr = -0.0001
    val caught = intercept[IllegalArgumentException] {
      estimator.setFactorsLr(factorsLr)
    }
  }

  it should "validate setBiasWd" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasWd = -1.6
    val caught = intercept[IllegalArgumentException] {
      estimator.setBiasWd(biasWd)
    }
  }

  it should "validate setLinearWd" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearWd = -2.2
    val caught = intercept[IllegalArgumentException] {
      estimator.setLinearWd(linearWd)
    }
  }

  it should "validate setFactorsWd" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsWd = -1e8
    val caught = intercept[IllegalArgumentException] {
      estimator.setFactorsWd(factorsWd)
    }
  }

  it should "validate setBiasInitMethod" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitMethod = "normally"
    val caught = intercept[IllegalArgumentException] {
      estimator.setBiasInitMethod(biasInitMethod)
    }
  }

  it should "validate setBiasInitScale" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitScale = -1.10
    val caught = intercept[IllegalArgumentException] {
      estimator.setBiasInitScale(biasInitScale)
    }
  }

  it should "validate setBiasInitSigma" in {
    val estimator = createFactorizationMachinesRegressor()
    val biasInitSigma = -0.0001
    val caught = intercept[IllegalArgumentException] {
      estimator.setBiasInitSigma(biasInitSigma)
    }
  }

  it should "validate setLinearInitMethod" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitMethod = "foo"
    val caught = intercept[IllegalArgumentException] {
      estimator.setLinearInitMethod(linearInitMethod)
    }
  }

  it should "validate setLinearInitScale" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitScale = -1e1
    val caught = intercept[IllegalArgumentException] {
      estimator.setLinearInitScale(linearInitScale)
    }
  }

  it should "validate setLinearInitSigma" in {
    val estimator = createFactorizationMachinesRegressor()
    val linearInitSigma = -1
    val caught = intercept[IllegalArgumentException] {
      estimator.setLinearInitSigma(linearInitSigma)
    }
  }

  it should "validate setFactorsInitMethod" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitMethod = "blah"
    val caught = intercept[IllegalArgumentException] {
      estimator.setFactorsInitMethod(factorsInitMethod)
    }
  }

  it should "validate setFactorsInitScale" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitScale = -10.10
    val caught = intercept[IllegalArgumentException] {
      estimator.setFactorsInitScale(factorsInitScale)
    }
  }

  it should "validate setFactorsInitSigma" in {
    val estimator = createFactorizationMachinesRegressor()
    val factorsInitSigma = -1.11
    val caught = intercept[IllegalArgumentException] {
      estimator.setFactorsInitSigma(factorsInitSigma)
    }
  }

  it should "throw on missing required parameter - none is set" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    val caught = intercept[NoSuchElementException] {
      estimator.transformSchema(null)
    }
  }

  it should "throw on missing required parameter - one is set" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    estimator.setFeatureDim(2)
    val caught = intercept[NoSuchElementException] {
      estimator.transformSchema(null)
    }
  }

  it should "throw on missing required parameter - two are set" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    val caught = intercept[NoSuchElementException] {
      estimator.transformSchema(null)
    }
  }

  it should "create full hyper-parameter map for binary classifier" in {
    val estimator = createFactorizationMachinesBinaryClassifier()
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    estimator.setNumFactors(3)
    estimator.setEpochs(2)
    estimator.setClipGradient(2.2)
    estimator.setEps(0.8)
    estimator.setRescaleGrad(0.8)
    estimator.setBiasLr(8.8)
    estimator.setLinearLr(5)
    estimator.setFactorsLr(1.1)
    estimator.setBiasWd(0.07)
    estimator.setLinearWd(0.01)
    estimator.setFactorsWd(0.2)
    estimator.setBiasInitMethod("uniform")
    estimator.setBiasInitScale(1.1)
    estimator.setBiasInitSigma(2.2)
    estimator.setBiasInitValue(100)
    estimator.setLinearInitMethod("normal")
    estimator.setLinearInitScale(1.1)
    estimator.setLinearInitSigma(2.2)
    estimator.setLinearInitValue(100)
    estimator.setFactorsInitMethod("constant")
    estimator.setFactorsInitScale(1.1)
    estimator.setFactorsInitSigma(2.2)
    estimator.setFactorsInitValue(100)

    val hyperParamMap = Map(
      "predictor_type" -> "binary_classifier",
      "feature_dim" -> "2",
      "mini_batch_size" -> "2",
      "num_factors" -> "3",
      "epochs" -> "2",
      "clip_gradient" -> "2.2",
      "eps" -> "0.8",
      "rescale_grad" -> "0.8",
      "bias_lr" -> "8.8",
      "linear_lr" -> "5.0",
      "factors_lr" -> "1.1",
      "bias_wd" -> "0.07",
      "linear_wd" -> "0.01",
      "factors_wd" -> "0.2",
      "bias_init_method" -> "uniform",
      "bias_init_scale" -> "1.1",
      "bias_init_sigma" -> "2.2",
      "bias_init_value" -> "100.0",
      "linear_init_method" -> "normal",
      "linear_init_scale" -> "1.1",
      "linear_init_sigma" -> "2.2",
      "linear_init_value" -> "100.0",
      "factors_init_method" -> "constant",
      "factors_init_scale" -> "1.1",
      "factors_init_sigma" ->"2.2",
      "factors_init_value" -> "100.0"
    )

    assert(estimator.makeHyperParameters() == hyperParamMap.asJava)
  }

  it should "create hyper-parameters map for regressor" in {
    val estimator = createFactorizationMachinesRegressor()

    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    estimator.setNumFactors(3)
    estimator.setEpochs(2)
    estimator.setClipGradient(2.2)
    estimator.setEps(0.8)
    estimator.setRescaleGrad(0.8)
    estimator.setBiasLr(8.8)

    val hyperParamMap = Map(
      "predictor_type" -> "regressor",
      "feature_dim" -> "2",
      "mini_batch_size" -> "2",
      "num_factors" -> "3",
      "epochs" -> "2",
      "clip_gradient" -> "2.2",
      "eps" -> "0.8",
      "rescale_grad" -> "0.8",
      "bias_lr" -> "8.8"
    )

    assert(estimator.makeHyperParameters() == hyperParamMap.asJava)
  }
}
