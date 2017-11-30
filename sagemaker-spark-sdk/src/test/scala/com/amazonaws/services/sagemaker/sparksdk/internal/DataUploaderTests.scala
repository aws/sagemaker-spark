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

package com.amazonaws.services.sagemaker.sparksdk.internal

import java.io.Writer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, LocatedFileStatus, Path, RemoteIterator}
import org.mockito.Matchers.any
import org.mockito.Mockito._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.SparkContext
import org.apache.spark.sql.{Dataset, SparkSession}

import com.amazonaws.services.sagemaker.sparksdk.S3DataPath

class DataUploaderTests extends FlatSpec with Matchers with MockitoSugar with BeforeAndAfter {

  val bucket = "bucket"
  val objectPath = "objectPath"

  var dataset: Dataset[String] = _
  var dataUploader: DataUploader = _
  var dataUploaderSpy: DataUploader = _
  var mockFileSystem: FileSystem = _
  var mockWriter: Writer = _
  var mockSparkSession: SparkSession = _
  var mockSparkContext: SparkContext = _
  var mockHadoopConf: Configuration = _

  before {
    dataset = mock[Dataset[String]]
    dataUploader = new DataUploader("csv", Map())
    dataUploaderSpy = spy(dataUploader)
    mockFileSystem = mock[FileSystem]
    mockWriter = mock[Writer]
    mockSparkSession = mock[SparkSession]
    mockSparkContext = mock[SparkContext]
    mockHadoopConf = mock[Configuration]
    when(dataset.sparkSession).thenReturn(mockSparkSession)
    when(mockSparkSession.sparkContext).thenReturn(mockSparkContext)
    when(mockSparkContext.hadoopConfiguration).thenReturn(mockHadoopConf)
  }

  it should "detect that the user is using EMRFS based on the filesystem class name" in {
    val className = "EmrFileSystem"
    when(mockHadoopConf.get("fs.s3.impl")).thenReturn(className)

    val usingEMRFS = dataUploader.usingEMRFS(dataset)

    assert(usingEMRFS)
  }

  it should "use s3:// if the user is using EMRFS" in {
    val onEmr = true
    val fsScheme = dataUploader.getFSScheme(onEmr)
    assert(fsScheme == "s3")
  }

  it should "use s3a:// if the user is not using EMRFS" in {
    val onEmr = false
    val fsScheme = dataUploader.getFSScheme(onEmr)
    assert(fsScheme == "s3a")
  }

  it should "detect that the user is not using EMRFS if the name is null" in {
    when(dataset.sparkSession).thenReturn(mockSparkSession)
    when(mockSparkSession.sparkContext).thenReturn(mockSparkContext)
    when(mockSparkContext.hadoopConfiguration).thenReturn(mockHadoopConf)
    val className = null
    when(mockHadoopConf.get("fs.s3.impl")).thenReturn(className)

    val usingEMRFS = dataUploader.usingEMRFS(dataset)

    assert(!usingEMRFS)
  }

  it should "detect that the user is not using EMRFS if the name is empty" in {
    when(dataset.sparkSession).thenReturn(mockSparkSession)
    when(mockSparkSession.sparkContext).thenReturn(mockSparkContext)
    when(mockSparkContext.hadoopConfiguration).thenReturn(mockHadoopConf)
    val className = ""
    when(mockHadoopConf.get("fs.s3.impl")).thenReturn(className)

    val usingEMRFS = dataUploader.usingEMRFS(dataset)

    assert(!usingEMRFS)
  }

  it should "detect that the user is not using EMRFS if the name is wrong" in {
    when(dataset.sparkSession).thenReturn(mockSparkSession)
    when(mockSparkSession.sparkContext).thenReturn(mockSparkContext)
    when(mockSparkContext.hadoopConfiguration).thenReturn(mockHadoopConf)
    val className = "SomeOtherFilesystem"
    when(mockHadoopConf.get("fs.s3.impl")).thenReturn(className)

    val usingEMRFS = dataUploader.usingEMRFS(dataset)

    assert(!usingEMRFS)
  }

  it should "upload a manifest file to S3 when running on EMRFS" in {
    val s3Path = S3DataPath(bucket, objectPath)
    doNothing().when(dataUploaderSpy).writeData(any[Dataset[_]], any[String])
    doReturn(true).when(dataUploaderSpy).usingEMRFS(dataset)
    val manifestKey = "manifestKey"
    val mockHadoopFS = mock[FileSystem]
    doReturn(mockHadoopFS).when(dataUploaderSpy).getHadoopFilesystem(any[Path], any[Dataset[_]])
    doReturn(manifestKey).when(dataUploaderSpy).writeAndUploadManifest(any[Path], any[FileSystem])

    val dataUploadResults = dataUploaderSpy.uploadData(s3Path, dataset)

    assert(dataUploadResults.s3DataPath.bucket == bucket)
    assert(dataUploadResults.s3DataPath.objectPath == manifestKey)
    val isManifestDataUploadResult = dataUploadResults match {
      case ManifestDataUploadResult(_) => true
      case _ => false
    }
    assert(isManifestDataUploadResult)
  }

  it should "not upload a manifest file to S3 if the filesystem implementation is not EMRFS" in {
    val s3Path = S3DataPath(bucket, objectPath)
    doNothing().when(dataUploaderSpy).writeData(any[Dataset[_]], any[String])
    doReturn(false).when(dataUploaderSpy).usingEMRFS(dataset)

    val dataUploadResults = dataUploaderSpy.uploadData(s3Path, dataset)

    assert(dataUploadResults.s3DataPath.bucket == bucket)
    assert(dataUploadResults.s3DataPath.objectPath == objectPath)
    verify(dataUploaderSpy, never()).writeAndUploadManifest(any[Path], any[FileSystem])
    val isObjectPrefixUploadResult = dataUploadResults match {
      case ObjectPrefixUploadResult(_) => true
      case _ => false
    }
    assert(isObjectPrefixUploadResult)
  }

  it should "return a manifest key after uploading a manifest file to S3" in {
    val hadoopFSMock = mock[FileSystem]
    val inputPath = new Path(s"s3a://$bucket/$objectPath")
    val dataUploader = new DataUploader("csv", Map())
    val dataUploaderSpy = spy(dataUploader)
    doNothing().when(hadoopFSMock).copyFromLocalFile(any[Boolean], any[Path], any[Path])
    doNothing().when(dataUploaderSpy).writeManifest(any[Path], any[FileSystem], any[Writer])

    val manifestKey = dataUploaderSpy.writeAndUploadManifest(inputPath, hadoopFSMock)

    val manifestKeySuffix = ".manifest.txt"
    assert(manifestKey.contains(s"$objectPath$manifestKeySuffix"))
  }

  it should "distinguish between uploaded files and hadoop special files" in {
    assert(dataUploader.isHadoopSpecialFile("_SUCCESS"))
    assert(dataUploader.isHadoopSpecialFile("$folder$"))
    assert(!dataUploader.isHadoopSpecialFile("part-123456"))
  }

  it should "write the files listed by the filesystem, except for hadoop files" in {
    val iterator = mock[RemoteIterator[LocatedFileStatus]]
    when(iterator.hasNext).thenReturn(true).thenReturn(true).thenReturn(true).thenReturn(false)
    val fileName = "s3://bucket/objectPath/notAHadoopSpecialFile"
    val hadoopSpecialFile = "s3://bucket/objectPath/$folder$"
    val filePath = new Path(fileName)
    val hadoopSpecialFilePath = new Path(hadoopSpecialFile)
    val firstMockLocatedFileStatus = mock[LocatedFileStatus]
    val secondMockLocatedFileStatus = mock[LocatedFileStatus]
    val hadoopMockLocatedFileStatus = mock[LocatedFileStatus]
    when(firstMockLocatedFileStatus.getPath).thenReturn(filePath)
    when(secondMockLocatedFileStatus.getPath).thenReturn(filePath)
    when(hadoopMockLocatedFileStatus.getPath).thenReturn(hadoopSpecialFilePath)
    when(iterator.next)
      .thenReturn(firstMockLocatedFileStatus)
      .thenReturn(secondMockLocatedFileStatus)
      .thenReturn(hadoopMockLocatedFileStatus)
    when(mockFileSystem.listFiles(any[Path], any[Boolean])).thenReturn(iterator)

    dataUploader.writeManifest(new Path("s3://bucket/objectPath/"), mockFileSystem, mockWriter)

    verify(mockWriter).write("[{\"prefix\": \"s3://bucket/objectPath/\"}, " +
      "\"notAHadoopSpecialFile\", \"notAHadoopSpecialFile\"]")
  }

}
