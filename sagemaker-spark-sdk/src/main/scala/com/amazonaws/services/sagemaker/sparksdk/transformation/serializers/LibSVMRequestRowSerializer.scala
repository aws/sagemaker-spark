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

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.sparksdk.transformation.{ContentTypes, RequestRowSerializer}

/**
  * Extracts a label column and features column from a Row and serializes as a LibSVM record.
  * Each Row must contain a Double column and a Vector column containing the label and features
  * respectively. Row field indexes for the label and features are obtained by looking up the
  * index of labelColumnName and featuresColumnName respectively
  * in the specified schema.
  *
  * A schema must be specified before [[RequestRowSerializer#serializeRow]] is invoked by a
  * client of this RequestRowSerializer. The schema is set either on instantiation of this
  * RequestRowSerializer or by
  * [[RequestRowSerializer#setSchema]].
  *
  * @param schema The schema of Rows being serialized. This parameter is Optional as the
  *               schema may not be known when this serializer is constructed.
  * @param labelColumnName The name of the label column
  * @param featuresColumnName the name of the features column
  */
class LibSVMRequestRowSerializer (val schema : Option[StructType] = None,
                                  val labelColumnName : String = "label",
                                  val featuresColumnName : String = "features")
  extends RequestRowSerializer with java.io.Serializable {

  private var labelFieldIndex : Int = _
  private var featuresFieldIndex : Int = _

  if(schema.isDefined) {
    setSchema(schema.get)
  }

  /**
    * Serializes an object to an Array of bytes for transformation by a SageMaker endpoint
    *
    * @param row The row to serialize
    * @return An Array[Byte]
    */
  override def serializeRow(row: Row): Array[Byte] = {
    val label = row.getAs[Double](labelFieldIndex)
    val features = row.getAs[Vector](featuresFieldIndex)
    LibSVMRequestRowSerializerUtils.serializeLabeledFeatureVector(label, features)
  }

  /**
    * Returns [[com.amazonaws.services.sagemaker.sparksdk.transformation.ContentTypes.TEXT_LIBSVM]]
    */
  override val contentType: String = ContentTypes.TEXT_LIBSVM

  /**
    * Validates that the specified schema contains a Double column with name labelColumnName and
    * a Vector column with name featuresColumnName.
    *
    * @throws java.lang.IllegalArgumentException if the specified schema is invalid
    */
  override def validateSchema(schema : StructType): Unit = {
    SchemaValidators.labeledSchemaValidator(schema, labelColumnName, featuresColumnName)
  }

  /**
    * @inheritdoc
    *
    * This method must be invoked before calling serializeRow if no schema was set when this
    * RequestRowSerializer was constructed.
    */
  override def setSchema(schema : StructType): Unit = {
    super.setSchema(schema)
    this.labelFieldIndex = schema.fieldIndex(labelColumnName)
    this.featuresFieldIndex = schema.fieldIndex(featuresColumnName)
  }
}
