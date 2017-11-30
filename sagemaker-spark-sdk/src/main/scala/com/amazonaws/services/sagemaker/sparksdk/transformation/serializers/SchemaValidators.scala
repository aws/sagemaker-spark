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

package com.amazonaws.services.sagemaker.sparksdk.transformation.serializers

import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.sql.types.{DoubleType, StructType}

private[serializers] object SchemaValidators {
  def labeledSchemaValidator(schema: StructType,
                             labelColumnName: String,
                             featuresColumnName: String): Unit = {
    if (
      !schema.exists(f => f.name == labelColumnName && f.dataType == DoubleType) ||
      !schema.exists(f => f.name == featuresColumnName && f.dataType == SQLDataTypes.VectorType)) {
      throw new IllegalArgumentException(s"Expecting schema with DoubleType column with name " +
        s"$labelColumnName and Vector column with name $featuresColumnName. Got ${schema.toString}")
    }
  }

  def unlabeledSchemaValidator(schema: StructType, featuresColumnName: String): Unit = {
    if (!schema.exists(f => f.name == featuresColumnName &&
      f.dataType == SQLDataTypes.VectorType)) {
      throw new IllegalArgumentException(
        s"Expecting schema with Vector column with name" +
        s" $featuresColumnName. Got ${schema.toString}")
    }
  }
}
