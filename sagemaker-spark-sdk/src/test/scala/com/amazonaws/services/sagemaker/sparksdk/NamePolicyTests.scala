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

package com.amazonaws.services.sagemaker.sparksdk;

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.mock.MockitoSugar

class NamePolicyTests extends FlatSpec with Matchers with MockitoSugar {

  "RandomNamePolicy" should "have correct names" in {
    val policy = RandomNamePolicy()
    assert(policy.trainingJobName.contains("trainingJob"))
    assert(policy.modelName.contains("model"))
    assert(policy.endpointConfigName.contains("endpointConfig"))
    assert(policy.endpointName.contains("endpoint"))
  }

  "RandomNamePolicy" should "have correct prefix" in {
    val policy = RandomNamePolicy("prefix")
    assert(policy.trainingJobName.startsWith("prefix"))
    assert(policy.modelName.startsWith("prefix"))
    assert(policy.endpointConfigName.startsWith("prefix"))
    assert(policy.endpointName.startsWith("prefix"))
  }

  "RandomNamePolicyFactory" should "return a random name policy with correct prefixes" in {
    val factory = new RandomNamePolicyFactory("prefix")
    val policy = factory.createNamePolicy
    assert(policy.isInstanceOf[RandomNamePolicy])
    assert(policy.trainingJobName.startsWith("prefix"))
    assert(policy.modelName.startsWith("prefix"))
    assert(policy.endpointConfigName.startsWith("prefix"))
    assert(policy.endpointName.startsWith("prefix"))
  }
}
