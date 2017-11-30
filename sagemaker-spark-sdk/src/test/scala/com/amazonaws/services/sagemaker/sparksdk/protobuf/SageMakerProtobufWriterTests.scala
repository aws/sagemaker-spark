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

package com.amazonaws.services.sagemaker.sparksdk.protobuf

import java.io._
import java.nio.file.{Files, Paths}
import java.util.ServiceLoader

import scala.collection.JavaConversions._

import aialgorithms.proto2.RecordProto2.Record
import org.apache.hadoop.mapreduce.TaskAttemptContext
import org.scalatest.{BeforeAndAfter, FlatSpec}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.protobuf.RecordIOOutputFormat.SageMakerProtobufRecordWriter

class SageMakerProtobufWriterTests extends FlatSpec with MockitoSugar with BeforeAndAfter {

  val tempFilePath : String = "tempFilePath"
  val label : Double = 1.0
  val values : Array[Double] = Array[Double](10, 20, 30)
  val vector : SparseVector = new SparseVector(values.length,
    (0 until values.length).toArray, values)
  val taskAttemptContext = mock[TaskAttemptContext]

  it should "find the sagemaker-pbr file format" in {
    val dataSourceRegisterLoader = ServiceLoader.load(classOf[DataSourceRegister])
    val iter = dataSourceRegisterLoader.iterator()
    var foundSageMaker = false
    while (iter.hasNext) {
      val fileFormatShortName = iter.next.shortName
      if (fileFormatShortName.equals("sagemaker")) {
        foundSageMaker = true
      }
    }
    assert(foundSageMaker)
  }

  it should "write a row" in {
    val label = 1.0
    val values = (1d to 1000000d by 1d).toArray
    runSerializationTest(label, values, 1)
  }

  it should "write two rows" in {
    val label = 1.0
    val values = (1d to 1000000d by 1d).toArray
    runSerializationTest(label, values, 2)
  }

  it should "write a row with an empty values array" in {
    val label = 1.0
    val values = Array[Double]()
    runSerializationTest(label, values, 1)
  }

  it should "write a row with a differently-named features column"  in {
    val schema = new StructType().add(StructField("label", DoubleType))
      .add(StructField("renamedFeaturesColumn", VectorType))
    val outputStream = new FileOutputStream(tempFilePath)
    val writer = new SageMakerProtobufWriter(tempFilePath, taskAttemptContext, schema,
      Map[String, String]("featuresColumnName" -> "renamedFeaturesColumn")) {
      override lazy val recordWriter = new SageMakerProtobufRecordWriter(outputStream)
    }

    val row = new GenericRowWithSchema(Array(label, vector), schema)

    writer.write(row)

    val recordIterator = getRecordIteratorFromFilePath(tempFilePath)
    val record = recordIterator.next
    assert(record.getFeaturesCount > 0)
    assert(record.getLabelCount > 0)
  }

  it should "fail to write a row with a differently-named features column without the " +
    "featuresColumnName option" in {
    val schema = new StructType().add(StructField("label", DoubleType))
      .add(StructField("renamedFeaturesColumn", VectorType))
    val outputStream = new FileOutputStream(tempFilePath)
    val writer = new SageMakerProtobufWriter(tempFilePath, taskAttemptContext, schema) {
      override lazy val recordWriter = new SageMakerProtobufRecordWriter(outputStream)
    }

    val row = new GenericRowWithSchema(Array(label, vector), schema)
    intercept[IllegalArgumentException] {
      writer.write(row)
    }
  }

  it should "write a row with a differently-named label column" in {
    val schema = new StructType().add(StructField("renamedLabelColumn", DoubleType))
      .add(StructField("features", VectorType))
    val outputStream = new FileOutputStream(tempFilePath)
    val writer = new SageMakerProtobufWriter(tempFilePath, taskAttemptContext, schema,
      Map[String, String]("labelColumnName" -> "renamedLabelColumn")) {
      override lazy val recordWriter = new SageMakerProtobufRecordWriter(outputStream)
    }

    val row = new GenericRowWithSchema(Array(label, vector), schema)
    writer.write(row)

    val recordIterator = getRecordIteratorFromFilePath(tempFilePath)
    val record = recordIterator.next
    assert(record.getFeaturesCount > 0)
    assert(record.getLabelCount > 0)
  }

  it should "not write a row with the label if given a renamed label column that is not in " +
    "the row" in {
    val schema = new StructType().add(StructField("label", DoubleType))
      .add(StructField("features", VectorType))
    val outputStream = new FileOutputStream(tempFilePath)
    val writer = new SageMakerProtobufWriter(tempFilePath, taskAttemptContext, schema,
      Map[String, String]("labelColumnName" -> "renamedLabelColumn")) {
      override lazy val recordWriter = new SageMakerProtobufRecordWriter(outputStream)
    }

    val row = new GenericRowWithSchema(Array(label, vector), schema)
    writer.write(row)

    val recordIterator = getRecordIteratorFromFilePath(tempFilePath)
    val record = recordIterator.next
    assert(record.getLabelCount == 0)
  }

  private def getRecordIteratorFromFilePath(filePath: String): Iterator[Record] = {
    val is = new FileInputStream(tempFilePath)
    val protobufInRecordIOBytes = Files.readAllBytes(Paths.get(tempFilePath))
    ProtobufConverter.recordIOByteArrayToProtobufs(protobufInRecordIOBytes)
  }

  private def validateRecord(recordIterator: Iterator[Record], label: Double,
                             values: Array[Double]): Unit = {
    while (recordIterator.hasNext) {
      val record = recordIterator.next
      assert(label == getLabel(record))
      for ((features, recordFeatures) <- getFeatures(record) zip values) {
        assert(features == recordFeatures)
      }
    }
  }

  private def getLabel(record: Record) : Float = {
    assert(record.getLabelCount == 1)
    record.getLabel(0).getValue.getFloat32Tensor.getValues(0)
  }

  private def getFeatures(record: Record) : java.util.List[java.lang.Float] = {
    assert(record.getFeaturesCount == 1)
    record.getFeatures(0).getValue.getFloat32Tensor.getValuesList
  }

  private def runSerializationTest(label: Double, values: Array[Double], rowCount: Integer):
    Unit = {
    val vector = new SparseVector(values.length, (0 until values.length).toArray, values)

    val schema = new StructType().add(StructField("label", DoubleType)).add(
      StructField("features", VectorType))
    val outputStream = new FileOutputStream(tempFilePath)
    val writer = new SageMakerProtobufWriter(tempFilePath, taskAttemptContext, schema) {
      override lazy val recordWriter = new SageMakerProtobufRecordWriter(outputStream)
    }

    val row = new GenericRowWithSchema(Array(label, vector), schema)
    for (i <- 0 until rowCount) {
      writer.write(row)
    }

    val recordIterator = getRecordIteratorFromFilePath(tempFilePath)
    validateRecord(recordIterator, label, values)
  }

  after {
    val tempFile = new File(tempFilePath)
    if (tempFile.exists) {
      tempFile.delete
    }
  }
}
