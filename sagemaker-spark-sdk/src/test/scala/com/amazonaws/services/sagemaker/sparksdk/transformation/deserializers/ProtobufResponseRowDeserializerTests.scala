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

package com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers

import aialgorithms.proto2.RecordProto2
import aialgorithms.proto2.RecordProto2._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.protobuf.ProtobufConverter

class ProtobufResponseRowDeserializerTests extends FlatSpec with Matchers with MockitoSugar
  with BeforeAndAfter {

  it should "deserialize output with one scalar" in {
    val scalar = RecordProto2.Record.newBuilder()
      .addLabel(MapEntry.newBuilder().setKey("scalar")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(10)))).build

    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("scalar", DoubleType))))
    val recordIOBytes = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(scalar.toByteArray)
    val records = deserializer.deserializeResponse(recordIOBytes)
    val row = records.next()
    assert(row.getAs[Double]("scalar") == 10)
  }

  it should "deserialize output with one dense vector" in {
    val denseVector = RecordProto2.Record.newBuilder()
      .addLabel(MapEntry.newBuilder().setKey("denseVector")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(10).addValues(11).addValues(12)))).build
    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("denseVector", VectorType))))
    val recordIOBytes = ProtobufConverter
      .byteArrayToRecordIOEncodedByteArray(denseVector.toByteArray)
    val records = deserializer.deserializeResponse(recordIOBytes)
    val row = records.next()
    assert(row.getAs[Vector]("denseVector") == new DenseVector(Array(10, 11, 12)))
  }

  it should "deserialize output with one sparse vector" in {
    val sparseVector = RecordProto2.Record.newBuilder()
      .addLabel(MapEntry.newBuilder().setKey("sparseVector")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addShape(10).addKeys(1).addKeys(2).addKeys(3)
            .addValues(10).addValues(11).addValues(12)))).build
    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("sparseVector", VectorType))))
    val recordIOBytes = ProtobufConverter
      .byteArrayToRecordIOEncodedByteArray(sparseVector.toByteArray)
    val records = deserializer.deserializeResponse(recordIOBytes)
    val row = records.next()
    assert(row.getAs[Vector]("sparseVector") ==
      new SparseVector(10, Array(1, 2, 3), Array(10, 11, 12)))
  }

  it should "deserialize output with two scalars" in {
    val twoScalars = RecordProto2.Record.newBuilder()
      .addLabel(MapEntry.newBuilder().setKey("scalar1")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(10))))
      .addLabel(MapEntry.newBuilder().setKey("scalar2")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(11)))).build
    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("scalar1", DoubleType),
      StructField("scalar2", DoubleType))))
    val recordIOBytes = ProtobufConverter.
      byteArrayToRecordIOEncodedByteArray(twoScalars.toByteArray)
    val records = deserializer.deserializeResponse(recordIOBytes)
    val row = records.next()
    assert(row.getAs[Double]("scalar1") == 10)
    assert(row.getAs[Double]("scalar2") == 11)
  }

  it should "fail to deserialize incorrectly encoded byte arrays" in {
    val byteArray = Array[Byte](1, 2, 3, 4)
    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("scalar1", DoubleType))))
    intercept[RuntimeException] {
      deserializer.deserializeResponse(byteArray)
    }
  }

  it should "fail to deserialize a record that has fewer entries than the schema has columns" in {
    val twoScalars = RecordProto2.Record.newBuilder()
      .addLabel(MapEntry.newBuilder().setKey("scalar1")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(10))))
      .addLabel(MapEntry.newBuilder().setKey("scalar2")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(11)))).build
    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("scalar1", DoubleType),
      StructField("scalar2", DoubleType), StructField("scalar3", DoubleType))))
    val recordIOBytes = ProtobufConverter
      .byteArrayToRecordIOEncodedByteArray(twoScalars.toByteArray)
    intercept[RuntimeException] {
      val records = deserializer.deserializeResponse(recordIOBytes)
    }
  }

  it should "successfully deserialize a record that has more entries than the " +
    "schema has columns" in {
    val twoScalars = RecordProto2.Record.newBuilder()
      .addLabel(MapEntry.newBuilder().setKey("scalar1")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(10))))
      .addLabel(MapEntry.newBuilder().setKey("scalar2")
        .setValue(Value.newBuilder().setFloat32Tensor(
          Float32Tensor.newBuilder().addValues(11)))).build
    val deserializer = new ProtobufResponseRowDeserializer(
      StructType(Array(StructField("scalar1", DoubleType))))
    val recordIOBytes = ProtobufConverter
      .byteArrayToRecordIOEncodedByteArray(twoScalars.toByteArray)
    val records = deserializer.deserializeResponse(recordIOBytes)
    val row = records.next()
    assert(row.getAs[Double]("scalar1") == 10)
  }

  it should "create a KMeans deserializer with the correct schema and fields" in {
    val kmeansDeserializer = new KMeansProtobufResponseRowDeserializer()
    val schemaFieldNames = kmeansDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("distance_to_cluster"))
    assert(schemaFieldNames.contains("closest_cluster"))
    val protobufKeys = kmeansDeserializer.protobufKeys.get
    assert(protobufKeys.contains("distance_to_cluster"))
    assert(protobufKeys.contains("closest_cluster"))
    assert(schemaFieldNames(0) == protobufKeys(0))
    assert(schemaFieldNames(1) == protobufKeys(1))
  }

  it should "create a PCA deserializer with the correct schema and fields" in {
    val pcaDeserializer = new PCAProtobufResponseRowDeserializer()
    val schemaFieldNames = pcaDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("projection"))
    val protobufKeys = pcaDeserializer.protobufKeys.get
    assert(protobufKeys.contains("projection"))
    assert(schemaFieldNames(0) == protobufKeys(0))
  }

  it should "create a linear learner regressor deserializer with the correct schema and fields" in {
    val regressorDeserializer = new LinearLearnerRegressorProtobufResponseRowDeserializer()
    val schemaFieldNames = regressorDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("score"))
    val protobufKeys = regressorDeserializer.protobufKeys.get
    assert(protobufKeys.contains("score"))
    assert(schemaFieldNames(0) == protobufKeys(0))
  }

  it should "create a linear learner binary classifier deserializer with " +
    "the correct schema and fields" in {
    val classifierDeserializer = new LinearLearnerBinaryClassifierProtobufResponseRowDeserializer()
    val schemaFieldNames = classifierDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("score"))
    assert(schemaFieldNames.contains("predicted_label"))
    val protobufKeys = classifierDeserializer.protobufKeys.get
    assert(protobufKeys.contains("score"))
    assert(protobufKeys.contains("predicted_label"))
    assert(schemaFieldNames(0) == protobufKeys(0))
    assert(schemaFieldNames(1) == protobufKeys(1))
  }

  it should "create a factorization machines regressor deserializer with " +
    "the correct schema and fields" in {
    val regressorDeserializer = new FactorizationMachinesRegressorDeserializer()
    val schemaFieldNames = regressorDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("score"))
    val protobufKeys = regressorDeserializer.protobufKeys.get
    assert(protobufKeys.contains("score"))
    assert(schemaFieldNames(0) == protobufKeys(0))
  }

  it should "create a factorization machines binary classifier deserializer with " +
    "the correct schema and fields" in {
    val classifierDeserializer =
      new FactorizationMachinesBinaryClassifierDeserializer()
    val schemaFieldNames = classifierDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("score"))
    assert(schemaFieldNames.contains("predicted_label"))
    val protobufKeys = classifierDeserializer.protobufKeys.get
    assert(protobufKeys.contains("score"))
    assert(protobufKeys.contains("predicted_label"))
    assert(schemaFieldNames(0) == protobufKeys(0))
    assert(schemaFieldNames(1) == protobufKeys(1))
  }

  it should "create a LDA deserializer with the correct schema and fields" in {
    val ldaDeserializer = new LDAProtobufResponseRowDeserializer()
    val schemaFieldNames = ldaDeserializer.schema.fieldNames
    assert(schemaFieldNames.contains("topic_mixture"))
    val protobufKeys = ldaDeserializer.protobufKeys.get
    assert(protobufKeys.contains("topic_mixture"))
    assert(schemaFieldNames(0) == protobufKeys(0))
  }
}
