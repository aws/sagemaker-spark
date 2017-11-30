# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#   http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import subprocess
import sagemaker_pyspark


def test_classpath_script_works():

    p = subprocess.Popen(["sagemakerpyspark-jars"], stdout=subprocess.PIPE)
    output, errors = p.communicate()

    jars = sagemaker_pyspark.classpath_jars()

    script_jars = output.decode('utf-8').split(",")
    assert len(jars) == len(script_jars)
    assert jars[0] == script_jars[0]


def test_classpath_script_can_use_separators():

    p = subprocess.Popen("sagemakerpyspark-jars :".split(), stdout=subprocess.PIPE)
    output, errors = p.communicate()

    jars = sagemaker_pyspark.classpath_jars()

    script_jars = output.decode('utf-8').split(":")
    assert len(jars) == len(script_jars)
    assert jars[0] == script_jars[0]


def test_classpath_script_can_provide_help():

    p = subprocess.Popen("sagemakerpyspark-jars --help".split(), stdout=subprocess.PIPE)
    output, errors = p.communicate()

    output = output.decode('utf-8')
    print(output)
    assert "usage" in output


def test_emr_classpath_works():

    p = subprocess.Popen("sagemakerpyspark-emr-jars :".split(), stdout=subprocess.PIPE)
    output, errors = p.communicate()

    script_jars = output.decode('utf-8').split(":")

    for path in script_jars:
        jar = os.path.basename(path).lower()
        # The only JARs required for EMR are sagemaker* and *aws*
        assert 'sagemaker' in jar or 'aws' in jar
