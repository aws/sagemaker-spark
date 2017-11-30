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

import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, SQLDataTypes}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.protobuf.ProtobufConverter

class ProtobufRequestRowSerializerTests extends FlatSpec with Matchers with MockitoSugar {

  val labelColumnName = "label"
  val featuresColumnName = "features"
  val schema = StructType(Array(StructField(labelColumnName, DoubleType), StructField(
    featuresColumnName, VectorType)))

  it should "serialize a dense vector" in {
    val vec = new DenseVector(Seq(10.0, -100.0, 2.0).toArray)
    val row = new GenericRowWithSchema(values = Seq(1.0, vec).toArray, schema = schema)
    val rrs = new ProtobufRequestRowSerializer(Some(schema))
    val protobuf = ProtobufConverter.rowToProtobuf(row, featuresColumnName, Option.empty)
    val serialized = rrs.serializeRow(row)
    val protobufIterator = ProtobufConverter.recordIOByteArrayToProtobufs(serialized)
    val protobufFromRecordIO = protobufIterator.next

    assert(!protobufIterator.hasNext)
    assert(protobuf.equals(protobufFromRecordIO))
  }

  it should "serialize a sparse vector" in {
    val vec = new SparseVector(100, Seq[Int](0, 10).toArray, Seq[Double](-100.0, 100.1).toArray)
    val row = new GenericRowWithSchema(values = Seq(1.0, vec).toArray, schema = schema)
    val rrs = new ProtobufRequestRowSerializer(Some(schema))
    val protobuf = ProtobufConverter.rowToProtobuf(row, featuresColumnName, Option.empty)
    val serialized = rrs.serializeRow(row)
    val protobufIterator = ProtobufConverter.recordIOByteArrayToProtobufs(serialized)
    val protobufFromRecordIO = protobufIterator.next

    assert(!protobufIterator.hasNext)
    assert(protobuf.equals(protobufFromRecordIO))
  }

  it should "fail to set schema on invalid features name" in {
    val vec = new SparseVector(100, Seq[Int](0, 10).toArray, Seq[Double](-100.0, 100.1).toArray)
    val row = new GenericRowWithSchema(values = Seq(1.0, vec).toArray, schema = schema)
    intercept[IllegalArgumentException] {
      val rrs = new ProtobufRequestRowSerializer(Some(schema), featuresColumnName = "doesNotExist")
    }
  }


  it should "fail on invalid types" in {
    val schemaWithInvalidFeaturesType = StructType(Array(
      StructField("label", DoubleType, nullable = false),
      StructField("features", StringType, nullable = false)))
    intercept[RuntimeException] {
      new ProtobufRequestRowSerializer(Some(schemaWithInvalidFeaturesType))
    }
  }

  it should "validate correct schema" in {
    val validSchema = StructType(Array(
      StructField("features", SQLDataTypes.VectorType, nullable = false)))
    new ProtobufRequestRowSerializer(Some(validSchema))
  }
}
