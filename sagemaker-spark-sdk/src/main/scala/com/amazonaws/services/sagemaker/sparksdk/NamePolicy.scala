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

package com.amazonaws.services.sagemaker.sparksdk

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.util.Identifiable

/**
  * Creates a [[NamePolicy]] upon a call to [[NamePolicyFactory#createNamePolicy]]
  */
abstract class NamePolicyFactory {
  def createNamePolicy : NamePolicy
}

/**
  * Creates a [[RandomNamePolicy]] upon a call to [[RandomNamePolicyFactory#createNamePolicy]]
  *
  * @param prefix The common name prefix for all SageMaker entities named with this NamePolicy
  */
class RandomNamePolicyFactory(val prefix: String = "") extends NamePolicyFactory {
  override def createNamePolicy: NamePolicy = { new RandomNamePolicy(prefix)}
}

/**
  * Creates a [[CustomNamePolicy]] upon a call to [[CustomNamePolicyFactory#createNamePolicy]]
  *
  * @param trainingJobName The job name of the SageMaker entity with this NamePolicy
  * @param modelName The model name of the SageMaker entity with this NamePolicy
  * @param endpointConfigName The endpoint config name of the SageMaker entity with this NamePolicy
  * @param endpointName The endpoint name of the SageMaker entity with this NamePolicy
  */
class CustomNamePolicyFactory(trainingJobName: String,
                              modelName: String,
                              endpointConfigName: String,
                              endpointName: String) extends NamePolicyFactory {

  override def createNamePolicy: NamePolicy = {
    new CustomNamePolicy(trainingJobName, modelName, endpointConfigName, endpointName)
  }

}


/**
  * Creates a [[CustomNamePolicyWithTimeStampSuffixFactory]] upon a call to
  * [[CustomNamePolicyWithTimeStampSuffix#createNamePolicy]]
  *
  * @param trainingJobName The job name of the SageMaker entity with this NamePolicy
  * @param modelName The model name of the SageMaker entity with this NamePolicy
  * @param endpointConfigName The endpoint config name of the SageMaker entity with this NamePolicy
  * @param endpointName The endpoint name of the SageMaker entity with this NamePolicy
  *
  */
class CustomNamePolicyWithTimeStampSuffixFactory(trainingJobName: String,
                                                 modelName: String,
                                                 endpointConfigName: String,
                                                 endpointName: String) extends NamePolicyFactory {
  override def createNamePolicy: NamePolicy = {
    new CustomNamePolicyWithTimeStampSuffix(trainingJobName, modelName,
                                            endpointConfigName, endpointName)
  }
}

/**
  * Provides names for SageMaker entities created during fit in
  * [[com.amazonaws.services.sagemaker.sparksdk.SageMakerEstimator]]
  */
abstract class NamePolicy {
  val trainingJobName: String
  val modelName : String
  val endpointConfigName : String
  val endpointName : String
}

/**
  * Provides random, unique SageMaker entity names that begin with the specified prefix.
  *
  * @param prefix The common name prefix for all SageMaker entities named with this NamePolicy
  */
case class RandomNamePolicy(val prefix : String = "") extends NamePolicy {
  // To pass the training job name validation regex: ^[a-zA-Z0-9](-*[a-zA-Z0-9])*
  val uid = Identifiable.randomUID("").stripPrefix("_").replace("_", "-")

  private val timestamp : String =
    LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME).replace(":", "-").replace(".", "-")

  val trainingJobName = s"${prefix}trainingJob-$uid-$timestamp"
  val modelName = s"${prefix}model-$uid-$timestamp"
  val endpointConfigName = s"${prefix}endpointConfig-$uid-$timestamp"
  val endpointName = s"${prefix}endpoint-$uid-$timestamp"
}


/**
  * Provides user sepecified SageMaker entity names.
  *
  * @param trainingJobName The job name of the SageMaker entity with this NamePolicy
  * @param modelName The model name of the SageMaker entity with this NamePolicy
  * @param endpointConfigName The endpoint config name of the SageMaker entity with this NamePolicy
  * @param endpointName The endpoint name of the SageMaker entity with this NamePolicy
  */
case class CustomNamePolicy(trainingJobName: String, modelName: String,
                            endpointConfigName: String, endpointName: String) extends NamePolicy



/**
  * Provides user sepecified SageMaker entity names with a common timestamp suffix.
  *
  * @param _trainingJobName The job name of the SageMaker entity with this NamePolicy
  * @param _modelName The model name of the SageMaker entity with this NamePolicy
  * @param _endpointConfigName The endpoint config name of the SageMaker entity with this NamePolicy
  * @param _endpointName The endpoint name of the SageMaker entity with this NamePolicy
  */
case class CustomNamePolicyWithTimeStampSuffix(private val _trainingJobName: String,
                                               private val _modelName: String,
                                               private val _endpointConfigName: String,
                                               private val _endpointName: String
                                              ) extends NamePolicy {
  private val timestamp : String =
    LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME).replace(":", "-").replace(".", "-")

  val trainingJobName = s"${_trainingJobName}-$timestamp"
  val modelName = s"${_modelName}-$timestamp"
  val endpointConfigName = s"${_endpointConfigName}-$timestamp"
  val endpointName = s"${_endpointName}-$timestamp"
}
