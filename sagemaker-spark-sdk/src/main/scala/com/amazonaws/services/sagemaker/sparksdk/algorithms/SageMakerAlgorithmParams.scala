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

package com.amazonaws.services.sagemaker.sparksdk.algorithms

import org.apache.spark.ml.param.{IntParam, Param, Params, ParamValidators}

/**
  * Params shared across first-party algorithms
  */
private[algorithms] trait SageMakerAlgorithmParams extends Params {
  /**
    * The number of examples in a mini-batch. Must be > 0.
    * Required.
    */
  val miniBatchSize : IntParam = new IntParam(this, "mini_batch_size",
    "The number of examples in a mini-batch. Must be > 0.", ParamValidators.gtEq(1))
  def getMiniBatchSize: Int = $(miniBatchSize)

  /**
    * The dimension of the input vectors. Must be > 0.
    * Required.
    *
    */
  val featureDim : IntParam = new IntParam(this, "feature_dim",
    "The dimension of the input vectors. Must be > 0.", ParamValidators.gtEq(1))
  def getFeatureDim: Int = $(featureDim)

  protected def autoOrAboveParamValidator(lowerBound: Double,
                                          inclusive: Boolean): String => Boolean = {
    (value: String) =>
      try {
        value == "auto" || {
          if (inclusive) {
            value.toDouble >= lowerBound
          }
          else {
            value.toDouble > lowerBound
          }
        }
      } catch {
        case e: NumberFormatException => false
      }
  }

  protected def inArrayOrAboveParamValidator(validValues: Array[String],
                                             lowerBound: Double): String => Boolean = {
    (value: String) =>
      try {
        validValues.contains(value) || {
          value.toDouble > lowerBound
        }
      } catch {
        case e: NumberFormatException => false
      }
  }

  protected def parseTrueAndFalse(param: Param[String]): Boolean = {
    $(param) match {
      case "True" => true
      case "False" => false
      case _ => throw new IllegalArgumentException("Param is neither 'True' nor 'False'")
    }
  }
}
