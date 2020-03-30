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
import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.mockito.MockitoSugar

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.KMeansProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

class KMeansSageMakerEstimatorTests extends FlatSpec with Matchers with MockitoSugar {

  def createKMeansEstimator(region: String = Regions.US_WEST_2.getName):
    KMeansSageMakerEstimator = {
    new KMeansSageMakerEstimator(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  it should "construct an estimator without a role" in {
    new KMeansSageMakerEstimator(
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1)
  }

  it should "use the correct defaults" in {
    val estimator = createKMeansEstimator()
    assert(estimator.trainingSparkDataFormat == "sagemaker")
    assert(estimator.requestRowSerializer.isInstanceOf[ProtobufRequestRowSerializer])
    assert(estimator.responseRowDeserializer.isInstanceOf[KMeansProtobufResponseRowDeserializer])
  }

  it should "use the correct images in all regions" in {
    val estimatorUSEast1 = createKMeansEstimator(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:1")

    val estimatorUSEast2 = createKMeansEstimator(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "404615174143.dkr.ecr.us-east-2.amazonaws.com/kmeans:1")

    val estimatorEUWest1 = createKMeansEstimator(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage
      == "438346466558.dkr.ecr.eu-west-1.amazonaws.com/kmeans:1")

    val estimatorUSWest2 = createKMeansEstimator(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage
      == "174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:1")

    val estimatorAPNorthEast1 = createKMeansEstimator(region = Regions.AP_NORTHEAST_1.getName)
    assert(estimatorAPNorthEast1.trainingImage
      == "351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/kmeans:1")

    val estimatorAPNorthEast2 = createKMeansEstimator(region = Regions.AP_NORTHEAST_2.getName)
    assert(estimatorAPNorthEast2.trainingImage ==
      "835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/kmeans:1")

    val estimatorEUCentral1 = createKMeansEstimator(region = Regions.EU_CENTRAL_1.getName)
    assert(estimatorEUCentral1.trainingImage ==
      "664544806723.dkr.ecr.eu-central-1.amazonaws.com/kmeans:1")

    val estimatorAPSouthEast2 = createKMeansEstimator(region = Regions.AP_SOUTHEAST_2.getName)
    assert(estimatorAPSouthEast2.trainingImage ==
      "712309505854.dkr.ecr.ap-southeast-2.amazonaws.com/kmeans:1")

    val estimatorGovCloud = createKMeansEstimator(region = Regions.GovCloud.getName)
    assert(estimatorGovCloud.trainingImage ==
      "226302683700.dkr.ecr.us-gov-west-1.amazonaws.com/kmeans:1")

    val estimatorAPSouth1 = createKMeansEstimator(region = Regions.AP_SOUTH_1.getName)
    assert(estimatorAPSouth1.trainingImage ==
      "991648021394.dkr.ecr.ap-south-1.amazonaws.com/kmeans:1")

    val estimatorAPSouthEast1 = createKMeansEstimator(region = Regions.AP_SOUTHEAST_1.getName)
    assert(estimatorAPSouthEast1.trainingImage ==
      "475088953585.dkr.ecr.ap-southeast-1.amazonaws.com/kmeans:1")

    val estimatorEUWest2 = createKMeansEstimator(region = Regions.EU_WEST_2.getName)
    assert(estimatorEUWest2.trainingImage ==
      "644912444149.dkr.ecr.eu-west-2.amazonaws.com/kmeans:1")

    val estimatorCACentral1 = createKMeansEstimator(region = Regions.CA_CENTRAL_1.getName)
    assert(estimatorCACentral1.trainingImage ==
      "469771592824.dkr.ecr.ca-central-1.amazonaws.com/kmeans:1")

    val estimatorUSWest1 = createKMeansEstimator(region = Regions.US_WEST_1.getName)
    assert(estimatorUSWest1.trainingImage ==
      "632365934929.dkr.ecr.us-west-1.amazonaws.com/kmeans:1")

    val estimatorAPEast1 = createKMeansEstimator(region = Regions.AP_EAST_1.getName)
    assert(estimatorAPEast1.trainingImage ==
      "286214385809.dkr.ecr.ap-east-1.amazonaws.com/kmeans:1")

    val estimatorSAEast1 = createKMeansEstimator(region = Regions.SA_EAST_1.getName)
    assert(estimatorSAEast1.trainingImage ==
      "855470959533.dkr.ecr.sa-east-1.amazonaws.com/kmeans:1")

    val estimatorEUNorth1 = createKMeansEstimator(region = Regions.EU_NORTH_1.getName)
    assert(estimatorEUNorth1.trainingImage ==
      "669576153137.dkr.ecr.eu-north-1.amazonaws.com/kmeans:1")

    val estimatorEUWest3 = createKMeansEstimator(region = Regions.EU_WEST_3.getName)
    assert(estimatorEUWest3.trainingImage ==
      "749696950732.dkr.ecr.eu-west-3.amazonaws.com/kmeans:1")

    val estimatorMESouth1 = createKMeansEstimator(region = Regions.ME_SOUTH_1.getName)
    assert(estimatorMESouth1.trainingImage ==
      "249704162688.dkr.ecr.me-south-1.amazonaws.com/kmeans:1")

    val estimatorCNNorth1 = createKMeansEstimator(region = Regions.CN_NORTH_1.getName)
    assert(estimatorCNNorth1.trainingImage ==
      "390948362332.dkr.ecr.cn-north-1.amazonaws.com.cn/kmeans:1")

    val estimatorCNNorthWest1 = createKMeansEstimator(region = Regions.CN_NORTHWEST_1.getName)
    assert(estimatorCNNorthWest1.trainingImage ==
      "387376663083.dkr.ecr.cn-northwest-1.amazonaws.com.cn/kmeans:1")
  }

  it should "setK" in {
    val estimator = createKMeansEstimator()
    val k = 100
    estimator.setK(k)
    assert(estimator.getK == k)
  }

  it should "setFeatureDim" in {
    val estimator = createKMeansEstimator()
    val featureDim = 100
    estimator.setFeatureDim(featureDim)
    assert(estimator.getFeatureDim == featureDim)
  }

  it should "setMiniBatchSize" in {
    val estimator = createKMeansEstimator()
    val miniBatchSize = 20
    estimator.setMiniBatchSize(miniBatchSize)
    assert(estimator.getMiniBatchSize == miniBatchSize)
  }

  it should "setMaxIter" in {
    val estimator = createKMeansEstimator()
    val maxIter = 20
    estimator.setMaxIter(maxIter)
    assert(estimator.getMaxIter == maxIter)
  }

  it should "setTol" in {
    val estimator = createKMeansEstimator()
    val tol = 0.1
    estimator.setTol(tol)
    assert(estimator.getTol == tol)
  }

  it should "setInitMethod" in {
    val estimator = createKMeansEstimator()
    val initMethod = "kmeans++"
    estimator.setInitMethod(initMethod)
    assert(estimator.getInitMethod == initMethod)
  }

  it should "setLocalInitMethod" in {
    val estimator = createKMeansEstimator()
    val localInitMethod = "random"
    estimator.setLocalInitMethod(localInitMethod)
    assert(estimator.getLocalInitMethod == localInitMethod)
  }

  it should "setHalfLifeTime" in {
    val estimator = createKMeansEstimator()
    val halflifeTime = 5
    estimator.setHalflifeTime(halflifeTime)
    assert(estimator.getHalflifeTime == halflifeTime)
  }

  it should "setEpochs" in {
    val estimator = createKMeansEstimator()
    val epochs = 5
    estimator.setEpochs(epochs)
    assert(estimator.getEpochs == epochs)
  }

  it should "setCenterFactor with a string" in {
    val estimator = createKMeansEstimator()
    val centerFactor = "auto"
    estimator.setCenterFactor(centerFactor)
    assert(estimator.getCenterFactor == centerFactor)
  }

  it should "setCenterFactor with an int" in {
    val estimator = createKMeansEstimator()
    val centerFactor = 5
    estimator.setCenterFactor(centerFactor)
    assert(estimator.getCenterFactor == centerFactor.toString)
  }

  it should "setTrialNum with a string" in {
    val estimator = createKMeansEstimator()
    val trialNum = "auto"
    estimator.setTrialNum(trialNum)
    assert(estimator.getTrialNum == trialNum)
  }

  it should "setTrialNum with an int" in {
    val estimator = createKMeansEstimator()
    val trialNum = 5
    estimator.setTrialNum(trialNum)
    assert(estimator.getTrialNum == trialNum.toString)
  }

  it should "setEvalMetrics" in {
    val estimator = createKMeansEstimator()
    val evalMetrics = "msd, ssd"
    estimator.setEvalMetrics(evalMetrics)
    assert(estimator.getEvalMetrics == evalMetrics)
  }

  it should "create correct hyper-parameter map" in {
    val estimator = createKMeansEstimator()
    estimator.setK(2)
    estimator.setFeatureDim(2)
    estimator.setMiniBatchSize(2)
    estimator.setMaxIter(2)
    estimator.setTol(0.2)
    estimator.setLocalInitMethod("random")
    estimator.setHalflifeTime(2)
    estimator.setEpochs(2)
    estimator.setInitMethod("kmeans++")
    estimator.setCenterFactor(2)
    estimator.setTrialNum(2)
    estimator.setEvalMetrics("ssd, msd")

    val hyperParamMap = Map(
      "k" -> "2",
      "feature_dim" -> "2",
      "mini_batch_size" -> "2",
      "init_method" -> "kmeans++",
      "local_lloyd_max_iter" -> "2",
      "local_lloyd_tol" -> "0.2",
      "local_lloyd_num_trials" -> "2",
      "local_lloyd_init_method" -> "random",
      "half_life_time_size" -> "2",
      "epochs" -> "2",
      "extra_center_factor" -> "2",
      "eval_metrics" -> "[ssd, msd]"
    )

    assert(hyperParamMap.asJava == estimator.makeHyperParameters())
  }

  it should "throw an IllegalArgumentException when invalid parameters are set" in {
    val ke1 = createKMeansEstimator()
    val caughtK = intercept[IllegalArgumentException] {
      ke1.setK(-1)
    }
    val caughtFeatureDim = intercept[IllegalArgumentException] {
      ke1.setFeatureDim(-1)
    }
    val caughtMiniBatchSize = intercept[IllegalArgumentException] {
      ke1.setMiniBatchSize(-1)
    }
    val caughtMaxIter = intercept[IllegalArgumentException] {
      ke1.setMaxIter(-1)
    }
    val caughtTol = intercept[IllegalArgumentException] {
      ke1.setTol(-1.0)
    }
    val caughtLocalInitMode = intercept[IllegalArgumentException] {
      ke1.setLocalInitMethod("SomeInvalidInitMode")
    }
    val caughtHalflifeTime = intercept[IllegalArgumentException] {
      ke1.setHalflifeTime(-1)
    }
    val caughtEpochs = intercept[IllegalArgumentException] {
      ke1.setEpochs(-1)
    }
    val caughtInitMode = intercept[IllegalArgumentException] {
      ke1.setInitMethod("SomeInvalidInitMode")
    }
    val caughtCenterFactor = intercept[IllegalArgumentException] {
      ke1.setCenterFactor("-1")
    }
    val caughtCenterFactor2 = intercept[IllegalArgumentException] {
      ke1.setCenterFactor("SomeNonInteger")
    }
    val caughtTrialNum = intercept[IllegalArgumentException] {
      ke1.setTrialNum("-1")
    }
    val caughtTrialNum2 = intercept[IllegalArgumentException] {
      ke1.setTrialNum("SomeNonInteger")
    }
    val caughtEvalMetrics = intercept[IllegalArgumentException] {
      ke1.setEvalMetrics("msd, ssd, SomeInvalidMetrics")
    }
  }
}
