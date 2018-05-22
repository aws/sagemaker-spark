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

import com.amazonaws.regions.Regions

private[algorithms] object SageMakerImageURIProvider {

  def getImage(region: String, regionAccountMap: Map[String, String],
               algorithmName: String, algorithmTag: String): String = {
    val account = regionAccountMap.get(region)
    account match {
      case None => throw new RuntimeException(s"The region $region is not supported." +
        s"Supported Regions: ${regionAccountMap.keys.mkString(", ")}")
      case _ => s"${account.get}.dkr.ecr.${region}.amazonaws.com/${algorithmName}:${algorithmTag}"
    }
  }
}

private[algorithms] object SagerMakerRegionAccountMaps {
  // For KMeans, PCA, Linear Learner, FactorizationMachines
  val AlgorithmsAccountMap: Map[String, String] = Map(
    Regions.EU_WEST_1.getName -> "438346466558",
    Regions.US_EAST_1.getName -> "382416733822",
    Regions.US_EAST_2.getName -> "404615174143",
    Regions.US_WEST_2.getName -> "174872318107",
    Regions.AP_NORTHEAST_1.getName -> "351501993468"
  )

  // For LDA
  val LDAAccountMap: Map[String, String] = Map(
    Regions.EU_WEST_1.getName -> "999678624901",
    Regions.US_EAST_1.getName -> "766337827248",
    Regions.US_EAST_2.getName -> "999911452149",
    Regions.US_WEST_2.getName -> "266724342769",
    Regions.AP_NORTHEAST_1.getName -> "258307448986"
  )

  // For XGBoost
  val ApplicationsAccountMap: Map[String, String] = Map(
    Regions.EU_WEST_1.getName -> "685385470294",
    Regions.US_EAST_1.getName -> "811284229777",
    Regions.US_EAST_2.getName -> "825641698319",
    Regions.US_WEST_2.getName -> "433757028032",
    Regions.AP_NORTHEAST_1.getName -> "501404015308"
  )
}

