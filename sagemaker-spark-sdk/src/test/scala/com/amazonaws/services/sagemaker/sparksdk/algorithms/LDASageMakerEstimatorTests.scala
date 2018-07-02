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
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.LDAProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

class LDASageMakerEstimatorTests extends FlatSpec with MockitoSugar {
  def createLDAEstimator(region: String = Regions.US_WEST_2.getName):
  LDASageMakerEstimator = {
    new LDASageMakerEstimator(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  it should "use the correct defaults" in {
    val estimator = createLDAEstimator()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.
      isInstanceOf[LDAProtobufResponseRowDeserializer])
  }

  it should "use the correct images in all regions" in {
    val estimatorUSEast1 =
      createLDAEstimator(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "766337827248.dkr.ecr.us-east-1.amazonaws.com/lda:1")

    val estimatorUSEast2 =
      createLDAEstimator(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "999911452149.dkr.ecr.us-east-2.amazonaws.com/lda:1")

    val estimatorEUWest1 =
      createLDAEstimator(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage ==
      "999678624901.dkr.ecr.eu-west-1.amazonaws.com/lda:1")

    val estimatorUSWest2 =
      createLDAEstimator(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage ==
      "266724342769.dkr.ecr.us-west-2.amazonaws.com/lda:1")

    val estimatorAPNorthEast1 =
      createLDAEstimator(region = Regions.AP_NORTHEAST_1.getName)
    assert(estimatorAPNorthEast1.trainingImage ==
      "258307448986.dkr.ecr.ap-northeast-1.amazonaws.com/lda:1")

    val estimatorAPNorthEast2 =
      createLDAEstimator(region = Regions.AP_NORTHEAST_2.getName)
    assert(estimatorAPNorthEast2.trainingImage ==
      "293181348795.dkr.ecr.ap-northeast-2.amazonaws.com/lda:1")

    val estimatorEUCentral1 =
      createLDAEstimator(region = Regions.EU_CENTRAL_1.getName)
    assert(estimatorEUCentral1.trainingImage ==
      "353608530281.dkr.ecr.eu-central-1.amazonaws.com/lda:1")

    val estimatorAPSouthEast2 =
      createLDAEstimator(region = Regions.AP_SOUTHEAST_2.getName)
    assert(estimatorAPSouthEast2.trainingImage ==
      "297031611018.dkr.ecr.ap-southeast-2.amazonaws.com/lda:1")
  }

  it should "setFeatureDim" in {
    val estimator = createLDAEstimator()
    val featureDim = 100
    estimator.setFeatureDim(featureDim)
    assert(featureDim == estimator.getFeatureDim)
  }

  it should "setMiniBatchSize" in {
    val estimator = createLDAEstimator()
    val miniBatchSize = 100
    estimator.setMiniBatchSize(miniBatchSize)
    assert(miniBatchSize == estimator.getMiniBatchSize)
  }

  it should "setNumTopics" in {
    val estimator = createLDAEstimator()
    val numTopics = 10
    estimator.setNumTopics(numTopics)
    assert(numTopics == estimator.getNumTopics)
  }

  it should "setAlpha0" in {
    val estimator = createLDAEstimator()
    val alpha0 = 10.10
    estimator.setAlpha0(alpha0)
    assert(alpha0 == estimator.getAlpha0)
  }

  it should "setMaxRestarts" in {
    val estimator = createLDAEstimator()
    val maxRestarts = 11
    estimator.setMaxRestarts(maxRestarts)
    assert(maxRestarts == estimator.getMaxRestarts)
  }

  it should "setMaxIterations" in {
    val estimator = createLDAEstimator()
    val maxIterations = 2
    estimator.setMaxIterations(maxIterations)
    assert(maxIterations == estimator.getMaxIterations)
  }

  it should "setTol" in {
    val estimator = createLDAEstimator()
    val tol = 1.11
    estimator.setTol(tol)
    assert(tol == estimator.getTol)
  }

  it should "validate setFeatureDim" in {
    val estimator = createLDAEstimator()
    val featureDim = -1
    val caught = intercept[IllegalArgumentException] {
      estimator.setFeatureDim(featureDim)
    }
  }

  it should "validate setMiniBatchSize" in {
    val estimator = createLDAEstimator()
    val miniBatchSize = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setMiniBatchSize(miniBatchSize)
    }
  }

  it should "validate setNumTopics" in {
    val estimator = createLDAEstimator()
    val numTopics = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setNumTopics(numTopics)
    }
  }

  it should "validate setAlpha0" in {
    val estimator = createLDAEstimator()
    val alpha0 = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setAlpha0(alpha0)
    }
  }

  it should "validate setMaxRestarts" in {
    val estimator = createLDAEstimator()
    val maxRestarts = 0
    val caught = intercept[IllegalArgumentException] {
      estimator.setMaxRestarts(maxRestarts)
    }
  }

  it should "validate setMaxIterations" in {
    val estimator = createLDAEstimator()
    val maxIterations = -5
    val caught = intercept[IllegalArgumentException] {
      estimator.setMaxIterations(maxIterations)
    }
  }

  it should "validate setTol" in {
    val estimator = createLDAEstimator()
    val tol = -0.0001
    val caught = intercept[IllegalArgumentException] {
      estimator.setTol(tol)
    }
  }

  it should "throw on missing required parameter - none is set" in {
    val estimator = createLDAEstimator()
    val caught = intercept[NoSuchElementException] {
      estimator.transformSchema(null)
    }
  }

  it should "throw on missing required parameter - one is set" in {
    val estimator = createLDAEstimator()
    estimator.setFeatureDim(2)
    val caught = intercept[NoSuchElementException] {
      estimator.transformSchema(null)
    }
  }

  it should "throw on missing required parameter - two are set" in {
    val estimator = createLDAEstimator()
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    val caught = intercept[NoSuchElementException] {
      estimator.transformSchema(null)
    }
  }

  it should "create full hyper-parameter map" in {
    val estimator = createLDAEstimator()
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    estimator.setNumTopics(3)
    estimator.setAlpha0(2.2)
    estimator.setMaxRestarts(5)
    estimator.setMaxIterations(8)
    estimator.setTol(0.8)

    val hyperParamMap = Map(
      "feature_dim" -> "2",
      "mini_batch_size" -> "2",
      "num_topics" -> "3",
      "alpha0" -> "2.2",
      "max_restarts" -> "5",
      "max_iterations" -> "8",
      "tol" -> "0.8"
    )

    assert(estimator.makeHyperParameters() == hyperParamMap.asJava)
  }
}

