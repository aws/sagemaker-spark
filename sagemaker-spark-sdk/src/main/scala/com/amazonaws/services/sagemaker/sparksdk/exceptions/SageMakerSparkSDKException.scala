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

package com.amazonaws.services.sagemaker.sparksdk.exceptions

import com.amazonaws.services.sagemaker.sparksdk.{CreatedResources, SageMakerEstimator, SageMakerModel, SageMakerResourceCleanup}

/**
  * Thrown if any failures occur during the creation or use of a [[SageMakerEstimator]]
  * or [[SageMakerModel]].
  *
  * @param message Message describing the failure details.
  * @param cause Original throwable from the underlying failure.
  * @param createdResources References to any resources created by the [[SageMakerEstimator]]
  *                         or [[SageMakerModel]]. [[SageMakerResourceCleanup]] can be used to
  *                         clean these up.
  */
case class SageMakerSparkSDKException(message: String,
                                      cause: Throwable,
                                      createdResources: CreatedResources)
  extends RuntimeException(message, cause)

/**
  * Thrown if any failures occur during resource cleanup of SageMaker entities created
  * from operations of the [[SageMakerEstimator]] and [[SageMakerModel]].
  *
  * @param message Message describing the failure details.
  * @param cause Original throwable from the underlying failure.
  * @param createdResources References to any resources created by the [[SageMakerEstimator]]
  *                         or [[SageMakerModel]]. [[SageMakerResourceCleanup]] can be used to
  *                         retry clean up.
  */
case class SageMakerResourceCleanupException(message: String,
                                             cause: Throwable,
                                             createdResources: CreatedResources)
  extends RuntimeException(message, cause)
