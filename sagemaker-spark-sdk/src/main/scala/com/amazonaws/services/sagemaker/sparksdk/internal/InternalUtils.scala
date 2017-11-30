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

package com.amazonaws.services.sagemaker.sparksdk.internal

import com.amazonaws.AmazonWebServiceRequest

private[sparksdk] object InternalUtils  {

  val PACKAGE_VERSION = getClass.getPackage.getImplementationVersion
  val SCALA_VERSION = util.Properties.versionNumberString
  val OS_NAME = System.getProperty("os.name")
  val OS_VERSION = System.getProperty("os.version")
  val USER_AGENT = "AWS-SageMaker-Spark-SDK/" + PACKAGE_VERSION +
    " scala/" + SCALA_VERSION + " " + OS_NAME + "/" + OS_VERSION

  /**
    * Appends custom user-agent string.
    * @param request The request that needs to be modified.
    * @return The request with appended user-agent string.
    */
  def applyUserAgent(request: AmazonWebServiceRequest): AmazonWebServiceRequest = {
    request.getRequestClientOptions.appendUserAgent(USER_AGENT)
    request
  }

}
