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

import org.scalatest._
import org.scalatest.mock.MockitoSugar
import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql._

class LibSVMResponseRowDeserializerTests extends FlatSpec with Matchers with MockitoSugar {

  "LibSVMResponseRowDeserializer" should "deserialize a single record with a two features" in {
    val rrd = new LibSVMResponseRowDeserializer(3)
    val responseIterator =
      rrd.deserializeResponse(createLibSVMRecord(1, Array(1, 2), Array(1.0, 2.0)).getBytes)
    assert(responseIterator.next == Row(1, new SparseVector(3, Array(1, 2), Array(1.0, 2.0))))
  }

  it should "deserialize a single record with no values" in {
    val rrd = new LibSVMResponseRowDeserializer(0)
    val responseIterator = rrd.deserializeResponse(
      createLibSVMRecord(1, Seq[Int]().toArray, Seq[Double]().toArray).getBytes)
    assert(responseIterator.next ==
      Row(1, new SparseVector(0, Seq[Int]().toArray, Seq[Double]().toArray)))
  }

  it should "deserialize multiple records with multiple features" in {
    val dim = 100
    val rrd = new LibSVMResponseRowDeserializer(dim)
    val sb = new StringBuilder
    val rows = new ListBuffer[Row]
    for (i <- Range(0, dim)) {
      val label = i.asInstanceOf[Double]
      val indices = Range (0, i)
      val values = Range(0, i) map( a => (a - 10) * a) map (a => a.asInstanceOf[Double])
      sb ++= createLibSVMRecord(label, indices.toArray, values.toArray)
      rows += Row(label, new SparseVector(dim, indices.toArray, values.toArray))
      sb ++= "\n"
    }
    assert(List() ++ rrd.deserializeResponse(sb.mkString.getBytes) == rows.toList)
  }

  it should "throw on invalid dimension" in {
    intercept[IllegalArgumentException] {
      new LibSVMResponseRowDeserializer(-1)
    }
  }

  it should "fail on invalid label" in {
    val rrd = new LibSVMResponseRowDeserializer(3)
    intercept[RuntimeException] {
      val responseIterator = rrd.deserializeResponse("XXX 1:1".getBytes)
    }
  }

  it should "fail on invalid value" in {
    val rrd = new LibSVMResponseRowDeserializer(3)
    intercept[RuntimeException] {
      rrd.deserializeResponse("1.0 1:Elizabeth".getBytes)
    }
  }

  it should "fail on invalid index" in {
    val rrd = new LibSVMResponseRowDeserializer(3)
    intercept[RuntimeException] {
      rrd.deserializeResponse("1.0 BLAH:1.3421".getBytes)
    }
  }

  it should "fail on missing index" in {
    val rrd = new LibSVMResponseRowDeserializer(3)
    intercept[RuntimeException] {
      rrd.deserializeResponse("1.0 :1.3421".getBytes)
    }
  }

  it should "fail on missing value" in {
    val rrd = new LibSVMResponseRowDeserializer(3)
    intercept[RuntimeException] {
      rrd.deserializeResponse("1.0 1:".getBytes)
    }
  }

  it should "fail on index out of bounds" in {
    val rrd = new LibSVMResponseRowDeserializer(2)
    intercept[RuntimeException] {
      rrd.deserializeResponse("1.0 3:2.0".getBytes)
    }
  }

  private def createLibSVMRecord(label : Double, indices : Array[Int], values : Array[Double])
  : String = {
    val sb = new StringBuilder(label.toString)
    val x = indices zip values
    for((index, value) <- x) {
      sb ++= s" ${index + 1}:$value"
    }
    sb.mkString
  }
}
