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

package com.amazonaws.services.sagemaker.sparksdk

import java.net.URISyntaxException

import org.scalatest.FlatSpec

class S3DataPathTests extends FlatSpec {

  "S3DataPath" should "create correct S3 URI string" in {
    val generatedUriString = new S3DataPath("my-bucket", "my-object").toS3UriString
    assert(generatedUriString == "s3://my-bucket/my-object")
  }

  it should "fail on load from invalid uri string" in {
    intercept[IllegalArgumentException] {
      S3DataPath.fromS3URI("http://not-an-s3-uri")
    }
    intercept[URISyntaxException] {
      S3DataPath.fromS3URI("You know nothing about URIs, John Snow")
    }
  }
}
