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

/**
  * References an IAM Role.
  */
abstract class IAMRoleResource

/**
  * Specifies an IAM Role by a Spark configuration lookup.
  *
  * @param configKey The Spark configuration key to read the IAM role ARN or name from
  */
case class IAMRoleFromConfig(val configKey :
                             String = "com.amazonaws.services.sagemaker.sparksdk.sagemakerrole")
  extends IAMRoleResource

/**
  * Specifies an IAM Role
  *
  * @param role The IAM role ARN or name
  */
case class IAMRole (val role : String) extends IAMRoleResource
