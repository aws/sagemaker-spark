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

import java.io.ByteArrayOutputStream

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FSDataOutputStream, Path}
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapreduce.TaskAttemptContext
import org.mockito.Matchers.any
import org.mockito.Mockito.{verify, when}
import org.scalatest.{BeforeAndAfter, FlatSpec}
import org.scalatest.mock.MockitoSugar

import com.amazonaws.services.sagemaker.sparksdk.protobuf.RecordIOOutputFormat.SageMakerProtobufRecordWriter


class RecordIOOutputFormatTests extends FlatSpec with MockitoSugar with BeforeAndAfter {

  var sagemakerProtobufRecordWriter: SageMakerProtobufRecordWriter = _
  var mockOutputStream : FSDataOutputStream = _
  var byteArrayOutputStream: ByteArrayOutputStream = _
  var mockTaskAttemptContext: TaskAttemptContext = _
  var mockPath: Path = _
  var mockFileSystem: FileSystem = _

  before {
    byteArrayOutputStream = new ByteArrayOutputStream()
    mockOutputStream = mock[FSDataOutputStream]
    sagemakerProtobufRecordWriter = new SageMakerProtobufRecordWriter(mockOutputStream)
    mockTaskAttemptContext = mock[TaskAttemptContext]
    mockPath = mock[Path]
    mockFileSystem = mock[FileSystem]
  }

  it should "write an empty array of bytes" in {
    val bytesWritable = new BytesWritable(byteArrayOutputStream.toByteArray)

    val bytes = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(bytesWritable.getBytes)
    sagemakerProtobufRecordWriter.write(NullWritable.get(), bytesWritable)

    verify(mockOutputStream).write(bytes, 0, bytes.length)
  }


  it should "write an array of bytes" in {
    val byteArray = Array[Byte](0, 0, 0, 0)
    byteArrayOutputStream.write(byteArray)
    val bytesWritable = new BytesWritable(byteArrayOutputStream.toByteArray)
    val bytes = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(bytesWritable.getBytes)

    sagemakerProtobufRecordWriter.write(NullWritable.get(), bytesWritable)

    verify(mockOutputStream).write(bytes, 0, bytes.length)
  }

  it should "write an array of bytes, padding as necessary" in {
    byteArrayOutputStream.write(5)
    val bytesWritable = new BytesWritable(byteArrayOutputStream.toByteArray)
    val bytes = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(bytesWritable.getBytes)

    sagemakerProtobufRecordWriter.write(NullWritable.get(), bytesWritable)

    verify(mockOutputStream).write(bytes, 0, bytes.length)
  }

  it should "write an array of bytes, padding only as much as necessary" in {
    byteArrayOutputStream.write(Array[Byte](0, 0, 0, 0, 0))
    val bytesWritable = new BytesWritable(byteArrayOutputStream.toByteArray)
    val bytes = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(bytesWritable.getBytes)

    sagemakerProtobufRecordWriter.write(NullWritable.get(), bytesWritable)

    verify(mockOutputStream).write(bytes, 0, bytes.length)
  }

  it should "create a record writer from a FSDataOutputStream created by the filesystem" in {
    val mockTaskAttemptContext = mock[TaskAttemptContext]
    val mockPath = mock[Path]
    val mockFileSystem = mock[FileSystem]
    when(mockPath.getFileSystem(any[Configuration])).thenReturn(mockFileSystem)
    new RecordIOOutputFormat() {
      override def getDefaultWorkFile(context: TaskAttemptContext, extension: String): Path = {
        mockPath
      }
    }.getRecordWriter(mockTaskAttemptContext)
    verify(mockFileSystem).create(mockPath, true)

  }

}
