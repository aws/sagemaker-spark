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
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.scalatest.mockito.MockitoSugar

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.XGBoostCSVRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.LibSVMRequestRowSerializer

class XGBoostSageMakerEstimatorTests extends FlatSpec with Matchers with MockitoSugar with
  BeforeAndAfter {

  var estimator: XGBoostSageMakerEstimator = _


  before {
    estimator = createXGBoostEstimator()
  }

  def createXGBoostEstimator(region: String =
                             Regions.US_WEST_2.getName): XGBoostSageMakerEstimator = {
    new XGBoostSageMakerEstimator(sagemakerRole = IAMRole("role"),
      trainingInstanceType = "ml.c4.xlarge",
      trainingInstanceCount = 2,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      region = Some(region))
  }

  it should "use the correct defaults" in {
    assert(estimator.trainingSparkDataFormat == "libsvm")
    assert(estimator.requestRowSerializer.isInstanceOf[LibSVMRequestRowSerializer])
    assert(estimator.responseRowDeserializer.isInstanceOf[XGBoostCSVRowDeserializer])
  }

  it should "use the correct images in all regions" in {
    val estimatorUSEast1 = createXGBoostEstimator(region = Regions.US_EAST_1.getName)
    assert(estimatorUSEast1.trainingImage ==
      "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1")

    val estimatorUSEast2 = createXGBoostEstimator(region = Regions.US_EAST_2.getName)
    assert(estimatorUSEast2.trainingImage ==
      "825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:1")

    val estimatorEUWest1 = createXGBoostEstimator(region = Regions.EU_WEST_1.getName)
    assert(estimatorEUWest1.trainingImage ==
      "685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:1")

    val estimatorUSWest2 = createXGBoostEstimator(region = Regions.US_WEST_2.getName)
    assert(estimatorUSWest2.trainingImage ==
      "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1")

    val estimatorAPNorthEast1 = createXGBoostEstimator(region = Regions.AP_NORTHEAST_1.getName)
    assert(estimatorAPNorthEast1.trainingImage ==
      "501404015308.dkr.ecr.ap-northeast-1.amazonaws.com/xgboost:1")

    val estimatorAPNorthEast2 = createXGBoostEstimator(region = Regions.AP_NORTHEAST_2.getName)
    assert(estimatorAPNorthEast2.trainingImage ==
      "306986355934.dkr.ecr.ap-northeast-2.amazonaws.com/xgboost:1")

    val estimatorEUCentral1 = createXGBoostEstimator(region = Regions.EU_CENTRAL_1.getName)
    assert(estimatorEUCentral1.trainingImage ==
      "813361260812.dkr.ecr.eu-central-1.amazonaws.com/xgboost:1")

    val estimatorAPSouthEast2 = createXGBoostEstimator(region = Regions.AP_SOUTHEAST_2.getName)
    assert(estimatorAPSouthEast2.trainingImage ==
      "544295431143.dkr.ecr.ap-southeast-2.amazonaws.com/xgboost:1")

    val estimatorGovCloud = createXGBoostEstimator(region = Regions.GovCloud.getName)
    assert(estimatorGovCloud.trainingImage ==
      "226302683700.dkr.ecr.us-gov-west-1.amazonaws.com/xgboost:1")

    val estimatorAPSouth1 = createXGBoostEstimator(region = Regions.AP_SOUTH_1.getName)
    assert(estimatorAPSouth1.trainingImage ==
      "991648021394.dkr.ecr.ap-south-1.amazonaws.com/xgboost:1")

    val estimatorAPSouthEast1 = createXGBoostEstimator(region = Regions.AP_SOUTHEAST_1.getName)
    assert(estimatorAPSouthEast1.trainingImage ==
      "475088953585.dkr.ecr.ap-southeast-1.amazonaws.com/xgboost:1")

    val estimatorEUWest2 = createXGBoostEstimator(region = Regions.EU_WEST_2.getName)
    assert(estimatorEUWest2.trainingImage ==
      "644912444149.dkr.ecr.eu-west-2.amazonaws.com/xgboost:1")

    val estimatorCACentral1 = createXGBoostEstimator(region = Regions.CA_CENTRAL_1.getName)
    assert(estimatorCACentral1.trainingImage ==
      "469771592824.dkr.ecr.ca-central-1.amazonaws.com/xgboost:1")

    val estimatorUSWest1 = createXGBoostEstimator(region = Regions.US_WEST_1.getName)
    assert(estimatorUSWest1.trainingImage ==
      "632365934929.dkr.ecr.us-west-1.amazonaws.com/xgboost:1")
  }

  it should "setBooster" in {
    val booster = "gblinear"
    estimator.setBooster(booster)
    assert(booster == estimator.getBooster)
  }


  it should "setSilent" in {
    val silent = 1
    estimator.setSilent(silent)
    assert(silent == estimator.getSilent)
  }


  it should "setNThread" in {
    val nThread = 5
    estimator.setNThread(nThread)
    assert(nThread == estimator.getNThread)
  }


  it should "setEta" in {
    val eta = 0.5
    estimator.setEta(eta)
    assert(eta == estimator.getEta)
  }


  it should "setGamma" in {
    val gamma = 10.0
    estimator.setGamma(gamma)
    assert(gamma == estimator.getGamma)
  }


  it should "setMaxDepth" in {
    val maxDepth = 10
    estimator.setMaxDepth(maxDepth)
    assert(maxDepth == estimator.getMaxDepth)
  }


  it should "setMinChildWeight" in {
    val minChildWeight = 10.0
    estimator.setMinChildWeight(minChildWeight)
    assert(minChildWeight == estimator.getMinChildWeight)
  }


  it should "setMaxDeltaStep" in {
    val maxDeltaStep = 10.0
    estimator.setMaxDeltaStep(maxDeltaStep)
    assert(maxDeltaStep == estimator.getMaxDeltaStep)
  }


  it should "setSubsample" in {
    val subsample = 0.5
    estimator.setSubsample(subsample)
    assert(subsample == estimator.getSubsample)
  }


  it should "setColSampleByTree" in {
    val colSampleByTree = 0.5
    estimator.setColSampleByTree(colSampleByTree)
    assert(colSampleByTree == estimator.getColSampleByTree)
  }


  it should "setColSampleByLevel" in {
    val colSampleByLevel = 0.5
    estimator.setColSampleByLevel(colSampleByLevel)
    assert(colSampleByLevel == estimator.getColSampleByLevel)
  }


  it should "setLambda" in {
    val lambda = 10.0
    estimator.setLambda(lambda)
    assert(lambda == estimator.getLambda)
  }


  it should "setAlpha" in {
    val alpha = 10.0
    estimator.setAlpha(alpha)
    assert(alpha == estimator.getAlpha)
  }


  it should "setTreeMethod" in {
    val treeMethod = "exact"
    estimator.setTreeMethod(treeMethod)
    assert(treeMethod == estimator.getTreeMethod)
  }


  it should "setSketchEps" in {
    val sketchEps = 0.5
    estimator.setSketchEps(sketchEps)
    assert(sketchEps == estimator.getSketchEps)
  }


  it should "setScalePosWeight" in {
    val scalePosWeight = 10.0
    estimator.setScalePosWeight(scalePosWeight)
    assert(scalePosWeight == estimator.getScalePosWeight)
  }


  it should "setUpdater" in {
    val updater = "grow_histmaker, prune"
    estimator.setUpdater(updater)
    assert(updater == estimator.getUpdater)
  }


  it should "setRefreshLeaf" in {
    val refreshLeaf = 1
    estimator.setRefreshLeaf(refreshLeaf)
    assert(refreshLeaf == estimator.getRefreshLeaf)
  }


  it should "setProcessType" in {
    val processType = "update"
    estimator.setProcessType(processType)
    assert(processType == estimator.getProcessType)
  }


  it should "setGrowPolicy" in {
    val growPolicy = "lossguide"
    estimator.setGrowPolicy(growPolicy)
    assert(growPolicy == estimator.getGrowPolicy)
  }


  it should "setMaxLeaves" in {
    val maxLeaves = 10
    estimator.setMaxLeaves(maxLeaves)
    assert(maxLeaves == estimator.getMaxLeaves)
  }


  it should "setMaxBin" in {
    val maxBin = 10
    estimator.setMaxBin(maxBin)
    assert(maxBin == estimator.getMaxBin)
  }


  it should "setSampleType" in {
    val sampleType = "weighted"
    estimator.setSampleType(sampleType)
    assert(sampleType == estimator.getSampleType)
  }


  it should "setNormalizeType" in {
    val normalizeType = "tree"
    estimator.setNormalizeType(normalizeType)
    assert(normalizeType == estimator.getNormalizeType)
  }


  it should "setRateDrop" in {
    val rateDrop = 0.5
    estimator.setRateDrop(rateDrop)
    assert(rateDrop == estimator.getRateDrop)
  }


  it should "setOneDrop" in {
    val oneDrop = 1
    estimator.setOneDrop(oneDrop)
    assert(1 == estimator.getOneDrop)
  }


  it should "setSkipDrop" in {
    val skipDrop = 0.5
    estimator.setSkipDrop(skipDrop)
    assert(skipDrop == estimator.getSkipDrop)
  }


  it should "setLambdaBias" in {
    val lambdaBias = 0.5
    estimator.setLambdaBias(lambdaBias)
    assert(lambdaBias == estimator.getLambdaBias)
  }


  it should "setTweedieVariancePower" in {
    val tweedieVariancePower = 1.5
    estimator.setTweedieVariancePower(tweedieVariancePower)
    assert(tweedieVariancePower == estimator.getTweedieVariancePower)
  }


  it should "setObjective" in {
    val objective = "multi:softmax"
    estimator.setObjective(objective)
    assert(objective == estimator.getObjective)
  }


  it should "setNumClasses" in {
    val numClasses = 10
    estimator.setNumClasses(numClasses)
    assert(numClasses == estimator.getNumClasses)
  }


  it should "setBaseScore" in {
    val baseScore = 0.5
    estimator.setBaseScore(baseScore)
    assert(baseScore == estimator.getBaseScore)
  }


  it should "setEvalMetric" in {
    val evalMetric = "logloss"
    estimator.setEvalMetric(evalMetric)
    assert(evalMetric == estimator.getEvalMetric)
  }


  it should "setSeed" in {
    val seed = 10
    estimator.setSeed(seed)
    assert(seed == estimator.getSeed)
  }


  it should "setNumRound" in {
    val numRound = 10
    estimator.setNumRound(numRound)
    assert(numRound == estimator.getNumRound)
  }

  it should "create correct hyper-parameter map" in {
    estimator.setBooster("dart")
    estimator.setSilent(0)
    estimator.setNThread(2)
    estimator.setEta(0.3)
    estimator.setGamma(0.1)
    estimator.setMaxDepth(8)
    estimator.setMinChildWeight(0.8)
    estimator.setMaxDeltaStep(0.8)
    estimator.setSubsample(0.8)
    estimator.setColSampleByTree(0.8)
    estimator.setColSampleByLevel(0.8)
    estimator.setLambda(1)
    estimator.setAlpha(0)
    estimator.setTreeMethod("auto")
    estimator.setSketchEps(0.03)
    estimator.setScalePosWeight(1)
    estimator.setUpdater("sync")
    estimator.setRefreshLeaf(1)
    estimator.setProcessType("update")
    estimator.setGrowPolicy("lossguide")
    estimator.setMaxLeaves(0)
    estimator.setMaxBin(256)
    estimator.setSampleType("weighted")
    estimator.setNormalizeType("forest")
    estimator.setRateDrop(0.5)
    estimator.setOneDrop(0)
    estimator.setSkipDrop(0.5)
    estimator.setLambdaBias(0.5)
    estimator.setTweedieVariancePower(1.5)
    estimator.setObjective("reg:logistic")
    estimator.setBaseScore(0.5)
    estimator.setEvalMetric("mae")
    estimator.setSeed(0)

    val hyperParamMap = Map(
      "booster" -> "dart",
      "silent" -> "0",
      "nthread" -> "2",
      "eta" -> "0.3",
      "gamma" -> "0.1",
      "max_depth" -> "8",
      "min_child_weight" -> "0.8",
      "max_delta_step" -> "0.8",
      "subsample" -> "0.8",
      "colsample_bytree" -> "0.8",
      "colsample_bylevel" -> "0.8",
      "lambda" -> "1.0",
      "alpha" -> "0.0",
      "tree_method" -> "auto",
      "sketch_eps" -> "0.03",
      "scale_pos_weight" -> "1.0",
      "updater" -> "sync",
      "refresh_leaf" -> "1",
      "process_type" -> "update",
      "grow_policy" -> "lossguide",
      "max_leaves" -> "0",
      "max_bin" -> "256",
      "sample_type" -> "weighted",
      "normalize_type" -> "forest",
      "rate_drop" -> "0.5",
      "one_drop" -> "0",
      "skip_drop" -> "0.5",
      "lambda_bias" -> "0.5",
      "tweedie_variance_power" -> "1.5",
      "objective" -> "reg:logistic",
      "base_score" -> "0.5",
      "eval_metric" -> "mae",
      "seed" -> "0"
    )

    assert(hyperParamMap.asJava == estimator.makeHyperParameters())
  }


  it should "validate booster" in {
    val booster = intercept[IllegalArgumentException] {
      estimator.setBooster("notABooster")
    }
  }


  it should "validate nThread" in {
    val nThread = intercept[IllegalArgumentException] {
      estimator.setNThread(0)
    }
  }


  it should "validate eta" in {
    val eta = intercept[IllegalArgumentException] {
      estimator.setEta(1.5)
    }
  }


  it should "validate gamma" in {
    val gamma = intercept[IllegalArgumentException] {
      estimator.setGamma(-1.0)
    }
  }


  it should "validate maxDepth" in {
    val maxDepth = intercept[IllegalArgumentException] {
      estimator.setMaxDepth(-1)
    }
  }


  it should "validate minChildWeight" in {
    val minChildWeight = intercept[IllegalArgumentException] {
      estimator.setMinChildWeight(-1.0)
    }
  }


  it should "validate maxDeltaStep" in {
    val maxDeltaStep = intercept[IllegalArgumentException] {
      estimator.setMaxDeltaStep(-10.0)
    }
  }


  it should "validate subsample" in {
    val subsample = intercept[IllegalArgumentException] {
      estimator.setSubsample(0.0)
    }
  }


  it should "validate colSampleByTree" in {
    val colSampleByTree = intercept[IllegalArgumentException] {
      estimator.setColSampleByTree(0.0)
    }
  }


  it should "validate colSampleByLevel" in {
    val colSampleByLevel = intercept[IllegalArgumentException] {
      estimator.setColSampleByLevel(0.0)
    }
  }


  it should "validate treeMethod" in {
    val treeMethod = intercept[IllegalArgumentException] {
      estimator.setTreeMethod("notATreeMethod")
    }
  }


  it should "validate sketchEps" in {
    val sketchEps = intercept[IllegalArgumentException] {
      estimator.setSketchEps(1.5)
    }
  }


  it should "validate updater" in {
    val updater = intercept[IllegalArgumentException] {
      estimator.setUpdater("notAnUpdater,List")
    }
  }


  it should "validate processType" in {
    val processType = intercept[IllegalArgumentException] {
      estimator.setProcessType("notAProcessType")
    }
  }


  it should "validate growPolicy" in {
    val growPolicy = intercept[IllegalArgumentException] {
      estimator.setGrowPolicy("notAGrowPolicy")
    }
  }


  it should "validate maxLeaves" in {
    val maxLeaves = intercept[IllegalArgumentException] {
      estimator.setMaxLeaves(-1)
    }
  }


  it should "validate maxBin" in {
    val maxBin = intercept[IllegalArgumentException] {
      estimator.setMaxBin(0)
    }
  }


  it should "validate sampleType" in {
    val sampleType = intercept[IllegalArgumentException] {
      estimator.setSampleType("notASampleType")
    }
  }


  it should "validate normalizeType" in {
    val normalizeType = intercept[IllegalArgumentException] {
      estimator.setNormalizeType("notANormalizeType")
    }
  }


  it should "validate rateDrop" in {
    val rateDrop = intercept[IllegalArgumentException] {
      estimator.setRateDrop(1.5)
    }
  }


  it should "validate skipDrop" in {
    val skipDrop = intercept[IllegalArgumentException] {
      estimator.setSkipDrop(1.5)
    }
  }


  it should "validate lambdaBias" in {
    val lambdaBias = intercept[IllegalArgumentException] {
      estimator.setLambdaBias(-1.0)
    }
  }


  it should "validate tweedieVariancePower" in {
    val tweedieVariancePower = intercept[IllegalArgumentException] {
      estimator.setTweedieVariancePower(2.5)
    }
  }


  it should "validate objective" in {
    val objective = intercept[IllegalArgumentException] {
      estimator.setObjective("notAn:objective")
    }
  }


  it should "validate numClasses" in {
    val numClasses = intercept[IllegalArgumentException] {
      estimator.setNumClasses(0)
    }
  }


  it should "validate evalMetric" in {
    val evalMetric = intercept[IllegalArgumentException] {
      estimator.setEvalMetric("notAnEvalMetric")
    }
  }


  it should "validate numRound" in {
    val numRound = intercept[IllegalArgumentException] {
      estimator.setNumRound(0)
    }
  }
}
