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
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StringType, StructField, StructType}

class UnlabeledLibSVMRequestRowSerializerTests extends FlatSpec with Matchers with MockitoSugar {

  val schema = StructType(Array(StructField("features", SQLDataTypes.VectorType, nullable = false)))

  "UnlabeledLibSVMRequestRowSerializer" should "serialize sparse vector" in {
    val vec = new SparseVector(100, Seq[Int](0, 10).toArray, Seq[Double](-100.0, 100.1).toArray)
    val row = new GenericRowWithSchema(values = Seq(vec).toArray, schema = schema)
    val rrs = new UnlabeledLibSVMRequestRowSerializer()
    val serialized = new String(rrs.serializeRow(row))
    assert ("0.0 1:-100.0 11:100.1\n" == serialized)
  }

  it should "serialize dense vector" in {
    val vec = new DenseVector(Seq(10.0, -100.0, 2.0).toArray)
    val row = new GenericRowWithSchema(values = Seq(vec).toArray, schema = schema)
    val rrs = new UnlabeledLibSVMRequestRowSerializer()
    val serialized = new String(rrs.serializeRow(row))
    assert("0.0 1:10.0 2:-100.0 3:2.0\n" == serialized)
  }

  it should "fail on invalid features column name" in {
    val vec = new DenseVector(Seq(10.0, -100.0, 2.0).toArray)
    val row = new GenericRowWithSchema(values = Seq(1.0, vec).toArray, schema = schema)
    val rrs =
      new UnlabeledLibSVMRequestRowSerializer(featuresColumnName = "mangoes are not features")
    intercept[RuntimeException] {
      rrs.serializeRow(row)
    }
  }

  it should "fail on invalid features type" in {
    val vec = new DenseVector(Seq(10.0, -100.0, 2.0).toArray)
    val row =
      new GenericRowWithSchema(values = Seq(1.0, "FEATURESSSSSZ!1!").toArray, schema = schema)
    val rrs = new UnlabeledLibSVMRequestRowSerializer()
    intercept[RuntimeException] {
      rrs.serializeRow(row)
    }
  }


  it should "validate correct schema" in {
    val validSchema = StructType(Array(
      StructField("features", SQLDataTypes.VectorType, nullable = false)))

    val rrs = new UnlabeledLibSVMRequestRowSerializer(Some(validSchema))
  }

  it should "fail to validate incorrect schema" in {
    val invalidSchema = StructType(Array(
      StructField("features", StringType, nullable = false)))

    intercept[IllegalArgumentException] {
      new UnlabeledLibSVMRequestRowSerializer(Some(invalidSchema))
    }
  }
}
