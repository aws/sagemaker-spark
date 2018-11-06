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
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.PCAProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

class PCASageMakerEstimatorTests extends FlatSpec with MockitoSugar {

  def createPCAEstimator(region: String = Regions.US_WEST_2.getName): PCASageMakerEstimator = {
    new PCASageMakerEstimator(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  it should "use the correct images in all regions" in {
    val estimatorUSEast1 = createPCAEstimator(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage == "382416733822.dkr.ecr.us-east-1.amazonaws.com/pca:1")

    val estimatorUSEast2 = createPCAEstimator(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage == "404615174143.dkr.ecr.us-east-2.amazonaws.com/pca:1")

    val estimatorEUWest1 = createPCAEstimator(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage == "438346466558.dkr.ecr.eu-west-1.amazonaws.com/pca:1")

    val estimatorUSWest2 = createPCAEstimator(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage == "174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1")

    val estimatorAPNorthEast1 = createPCAEstimator(region = Regions.AP_NORTHEAST_1.getName)
    assert(estimatorAPNorthEast1.trainingImage ==
      "351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/pca:1")

    val estimatorAPNorthEast2 = createPCAEstimator(region = Regions.AP_NORTHEAST_2.getName)
    assert(estimatorAPNorthEast2.trainingImage ==
      "835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/pca:1")

    val estimatorEUCentral1 = createPCAEstimator(region = Regions.EU_CENTRAL_1.getName)
    assert(estimatorEUCentral1.trainingImage ==
      "664544806723.dkr.ecr.eu-central-1.amazonaws.com/pca:1")

    val estimatorAPSouthEast2 = createPCAEstimator(region = Regions.AP_SOUTHEAST_2.getName)
    assert(estimatorAPSouthEast2.trainingImage ==
      "712309505854.dkr.ecr.ap-southeast-2.amazonaws.com/pca:1")

    val estimatorGovCloud = createPCAEstimator(region = Regions.GovCloud.getName)
    assert(estimatorGovCloud.trainingImage ==
      "226302683700.dkr.ecr.us-gov-west-1.amazonaws.com/pca:1")

    val estimatorAPSouth1 = createPCAEstimator(region = Regions.AP_SOUTH_1.getName)
    assert(estimatorAPSouth1.trainingImage ==
      "991648021394.dkr.ecr.ap-south-1.amazonaws.com/pca:1")

    val estimatorAPSouthEast1 = createPCAEstimator(region = Regions.AP_SOUTHEAST_1.getName)
    assert(estimatorAPSouthEast1.trainingImage ==
      "475088953585.dkr.ecr.ap-southeast-1.amazonaws.com/pca:1")

    val estimatorEUWest2 = createPCAEstimator(region = Regions.EU_WEST_2.getName)
    assert(estimatorEUWest2.trainingImage ==
      "644912444149.dkr.ecr.eu-west-2.amazonaws.com/pca:1")

    val estimatorCACentral1 = createPCAEstimator(region = Regions.CA_CENTRAL_1.getName)
    assert(estimatorCACentral1.trainingImage ==
      "469771592824.dkr.ecr.ca-central-1.amazonaws.com/pca:1")

    val estimatorUSWest1 = createPCAEstimator(region = Regions.US_WEST_1.getName)
    assert(estimatorUSWest1.trainingImage ==
      "632365934929.dkr.ecr.us-west-1.amazonaws.com/pca:1")
  }

  it should "use the correct defaults" in {
    val estimator = createPCAEstimator()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.isInstanceOf[PCAProtobufResponseRowDeserializer])
  }

  it should "setNumComponents" in {
    val estimator = createPCAEstimator()
    val numComponents = 100
    estimator.setNumComponents(numComponents)
    assert(numComponents == estimator.getNumComponents)
  }

  it should "setAlgorithmMode" in {
    val estimator = createPCAEstimator()
    val algorithmMode = "randomized"
    estimator.setAlgorithmMode(algorithmMode)
    assert(algorithmMode == estimator.getAlgorithmMode)
  }

  it should "setSubtractMean" in {
    val estimator = createPCAEstimator()
    val subtractMean = true
    estimator.setSubtractMean(subtractMean)
    assert(subtractMean == estimator.getSubtractMean)
  }

  it should "setFeatureDim" in {
    val estimator = createPCAEstimator()
    val featureDim = 100
    estimator.setFeatureDim(featureDim)
    assert(featureDim == estimator.getFeatureDim)
  }

  it should "setMiniBatchSize" in {
    val estimator = createPCAEstimator()
    val miniBatchSize = 100
    estimator.setMiniBatchSize(miniBatchSize)
    assert(miniBatchSize == estimator.getMiniBatchSize)
  }

  it should "setExtraComponents" in {
    val estimator = createPCAEstimator()
    val extraComponents = 10
    estimator.setExtraComponents(extraComponents)
    assert(extraComponents == estimator.getExtraComponents)
  }

  it should "create correct hyper-parameter map" in {
    val estimator = createPCAEstimator()
    estimator.setNumComponents(2)
    estimator.setAlgorithmMode("regular")
    estimator.setSubtractMean(false)
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    estimator.setExtraComponents(2)

    val hyperParamMap = Map(
      "num_components" -> "2",
      "feature_dim" -> "2",
      "mini_batch_size" -> "2",
      "algorithm_mode" -> "regular",
      "subtract_mean" -> "False",
      "extra_components" -> "2"
    )

    assert(hyperParamMap.asJava == estimator.makeHyperParameters())
  }

  it should "validate numComponents" in {
    val estimator = createPCAEstimator()
    val caughtK = intercept[IllegalArgumentException] {
      estimator.setNumComponents(-1)
    }
  }

  it should "validate feature dim" in {
    val estimator = createPCAEstimator()
    val caughtFeatureDim = intercept[IllegalArgumentException] {
      estimator.setFeatureDim(-1)
    }
  }

  it should "validate miniBatchSize" in {
    val estimator = createPCAEstimator()
    val caughtMiniBatchSize = intercept[IllegalArgumentException] {
      estimator.setMiniBatchSize(-1)
    }
  }

  it should "validate algorithmMode" in {
    val estimator = createPCAEstimator()
    val caughtAlgorithmMode = intercept[IllegalArgumentException] {
      estimator.setAlgorithmMode("invalid algorithm mode")
    }
  }

  it should "validate extra components" in {
    val estimator = createPCAEstimator()
    val caughtExtraComponents = intercept[IllegalArgumentException] {
      estimator.setExtraComponents(0)
    }
  }
}
