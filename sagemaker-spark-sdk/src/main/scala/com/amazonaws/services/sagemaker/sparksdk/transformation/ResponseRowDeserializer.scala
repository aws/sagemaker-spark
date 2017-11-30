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

package com.amazonaws.services.sagemaker.sparksdk.transformation

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

/**
  * Deserializes a SageMaker endpoint response into a series of [[Row]]s.
  * Each deserialized Row conforms to [[ResponseRowDeserializer.schema]]
  *
  * @see [[com.amazonaws.services.sagemaker.sparksdk.SageMakerModel]]
  */
trait ResponseRowDeserializer extends Serializable {

  /**
    * Deserialize a SageMaker response to a series of rows.
    *
    * @param responseData The Array[Byte] containing the SageMaker response
    * @return An Iterator over deserialized response [[Row]]s
    */
  def deserializeResponse(responseData: Array[Byte]) : Iterator[Row]

  /**
    * The schema of each Row in [[ResponseRowDeserializer#deserializeResponse]]
    */
  val schema : StructType

  /**
    * The content-type(s) this deserializer accepts, encoded as a HTTP accepts header
    */
  val accepts : String
}
