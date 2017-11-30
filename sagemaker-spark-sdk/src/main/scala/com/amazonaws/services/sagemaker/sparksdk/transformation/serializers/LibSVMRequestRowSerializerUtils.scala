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

import scala.collection.mutable.StringBuilder

import org.apache.spark.ml.linalg.Vector

private[serializers]object LibSVMRequestRowSerializerUtils {

   def serializeLabeledFeatureVector(label : Double, features : Vector): Array[Byte] = {
    val sb = new StringBuilder(label.toString)
    features.foreachActive {
      case (index, value) =>
        sb ++= s" ${index + 1}:$value"
    }
    sb ++= "\n"
    sb.toString().getBytes
  }
}
