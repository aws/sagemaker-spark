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
    Regions.AP_NORTHEAST_1.getName -> "351501993468",
    Regions.AP_NORTHEAST_2.getName -> "835164637446",
    Regions.EU_CENTRAL_1.getName -> "664544806723",
    Regions.AP_SOUTHEAST_2.getName -> "712309505854",
    Regions.GovCloud.getName -> "226302683700",
    Regions.AP_SOUTH_1.getName -> "991648021394",
    Regions.AP_SOUTHEAST_1.getName -> "475088953585",
    Regions.CA_CENTRAL_1.getName -> "469771592824",
    Regions.EU_WEST_2.getName -> "644912444149",
    Regions.US_WEST_1.getName -> "632365934929",
    Regions.AP_EAST_1.getName -> "286214385809",
    Regions.SA_EAST_1.getName -> "855470959533",
    Regions.EU_NORTH_1.getName -> "669576153137",
    Regions.EU_WEST_3.getName -> "749696950732"
  )

  // For LDA
  val LDAAccountMap: Map[String, String] = Map(
    Regions.EU_WEST_1.getName -> "999678624901",
    Regions.US_EAST_1.getName -> "766337827248",
    Regions.US_EAST_2.getName -> "999911452149",
    Regions.US_WEST_2.getName -> "266724342769",
    Regions.AP_NORTHEAST_1.getName -> "258307448986",
    Regions.AP_NORTHEAST_2.getName -> "293181348795",
    Regions.EU_CENTRAL_1.getName -> "353608530281",
    Regions.AP_SOUTHEAST_2.getName -> "297031611018",
    Regions.GovCloud.getName -> "226302683700",
    Regions.AP_SOUTH_1.getName -> "991648021394",
    Regions.AP_SOUTHEAST_1.getName -> "475088953585",
    Regions.CA_CENTRAL_1.getName -> "469771592824",
    Regions.EU_WEST_2.getName -> "644912444149",
    Regions.US_WEST_1.getName -> "632365934929"
  )

  // For XGBoost
  val ApplicationsAccountMap: Map[String, String] = Map(
    Regions.EU_WEST_1.getName -> "685385470294",
    Regions.US_EAST_1.getName -> "811284229777",
    Regions.US_EAST_2.getName -> "825641698319",
    Regions.US_WEST_2.getName -> "433757028032",
    Regions.AP_NORTHEAST_1.getName -> "501404015308",
    Regions.AP_NORTHEAST_2.getName -> "306986355934",
    Regions.EU_CENTRAL_1.getName -> "813361260812",
    Regions.AP_SOUTHEAST_2.getName -> "544295431143",
    Regions.GovCloud.getName -> "226302683700",
    Regions.AP_SOUTH_1.getName -> "991648021394",
    Regions.AP_SOUTHEAST_1.getName -> "475088953585",
    Regions.CA_CENTRAL_1.getName -> "469771592824",
    Regions.EU_WEST_2.getName -> "644912444149",
    Regions.US_WEST_1.getName -> "632365934929",
    Regions.AP_EAST_1.getName -> "286214385809",
    Regions.SA_EAST_1.getName -> "855470959533",
    Regions.EU_NORTH_1.getName -> "669576153137",
    Regions.EU_WEST_3.getName -> "749696950732"
  )
}

