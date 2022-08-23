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

import java.io.{File, FileWriter}

import scala.jdk.CollectionConverters._

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.sql.SparkSession

import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.LibSVMResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.LibSVMRequestRowSerializer

class LibSVMTransformationLocalFunctionalTests extends FlatSpec with Matchers with MockitoSugar
  with BeforeAndAfter {

  val spark = SparkSession.builder
    .master("local")
    .appName("spark session")
    .getOrCreate()

  var libsvmDataFile : File = _
  val libsvmdata =
    "1.0 1:1.5 2:3.0 28:-39.935 55:0.01\n" +
      "0.0 2:3.0 28:-39.935 55:0.01\n" +
      "-1.0 23:-39.935 55:0.01\n" +
      "3.0 1:1.5 2:3.0"
  before {
    libsvmDataFile = File.createTempFile("temp", "temp")
    val fw = new FileWriter(libsvmDataFile)
    fw.write(libsvmdata)
    fw.close()
  }

  "LibSVMSerialization" should "serialize Spark loaded libsvm file to same contents" in {
    import spark.implicits._

    val df = spark.read.format("libsvm").load(libsvmDataFile.getPath)
    val libsvmSerializer = new LibSVMRequestRowSerializer(Some(df.schema))
    val result = df.map(row => new String(libsvmSerializer.serializeRow(row))).collect().mkString
    assert (libsvmdata.trim == result.trim)
  }

  "LibSVMDeserialization" should "deserialize serialized lib svm records" in {

    val libsvmdata =
      "1.0 1:1.5 2:3.0 28:-39.935 55:0.01\n" +
        "0.0 2:3.0 28:-39.935 55:0.01\n" +
        "-1.0 23:-39.935 55:0.01\n" +
        "3.0 1:1.5 2:3.0"

    val libsvmDeserializer = new LibSVMResponseRowDeserializer (55)
    val rowList = libsvmDeserializer.deserializeResponse(libsvmdata.getBytes).toBuffer.asJava
    val deserializedDataFrame = spark.createDataFrame(rowList, libsvmDeserializer.schema)
    val sparkProducedDataFrame = spark.read.format("libsvm").load(libsvmDataFile.getPath)

    val deserializedRows = deserializedDataFrame.collectAsList()
    val sparkRows = sparkProducedDataFrame.collectAsList()

    assert (deserializedRows == sparkRows)
  }
}
