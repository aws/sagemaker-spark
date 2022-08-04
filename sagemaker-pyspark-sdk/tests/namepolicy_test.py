import pytest
import os

from sagemaker_pyspark import (CustomNamePolicyFactory, classpath_jars,
                               CustomNamePolicyWithTimeStampSuffixFactory,
                               CustomNamePolicyWithTimeStampSuffix,
                               CustomNamePolicy)

from pyspark import SparkConf, SparkContext
from py4j.java_gateway import JavaObject


@pytest.fixture(autouse=True)
def with_spark_context():
    os.environ['SPARK_CLASSPATH'] = ":".join(classpath_jars())
    conf = (SparkConf()
            .set("spark.driver.extraClassPath", os.environ['SPARK_CLASSPATH']))

    if SparkContext._active_spark_context is None:
        SparkContext(conf=conf)

    yield SparkContext._active_spark_context

    # TearDown
    SparkContext.stop(SparkContext._active_spark_context)


def test_CustomNamePolicyFactory():
    policy_factory = CustomNamePolicyFactory("jobName", "modelname", "epconfig", "ep")
    java_obj = policy_factory._to_java()
    assert (isinstance(java_obj, JavaObject))
    assert (java_obj.getClass().getSimpleName() == "CustomNamePolicyFactory")
    policy_name = java_obj.createNamePolicy().getClass().getSimpleName()
    assert (policy_name == "CustomNamePolicy")


def test_CustomNamePolicyWithTimeStampSuffixFactory():
    policy_factory = CustomNamePolicyWithTimeStampSuffixFactory("jobName", "modelname",
                                                                "epconfig", "ep")
    java_obj = policy_factory._to_java()
    assert (isinstance(java_obj, JavaObject))
    assert (java_obj.getClass().getSimpleName() == "CustomNamePolicyWithTimeStampSuffixFactory")
    policy_name = java_obj.createNamePolicy().getClass().getSimpleName()
    assert (policy_name == "CustomNamePolicyWithTimeStampSuffix")


def test_CustomNamePolicyWithTimeStampSuffix():
    name_policy = CustomNamePolicyWithTimeStampSuffix("jobName", "modelname", "epconfig", "ep")
    assert (isinstance(name_policy._to_java(), JavaObject))
    assert (name_policy._call_java("trainingJobName") != "jobName")
    assert (name_policy._call_java("modelName") != "modelname")
    assert (name_policy._call_java("endpointConfigName") != "epconfig")
    assert (name_policy._call_java("endpointName") != "ep")

    assert (name_policy._call_java("trainingJobName").startswith("jobName"))
    assert (name_policy._call_java("modelName").startswith("modelname"))
    assert (name_policy._call_java("endpointConfigName").startswith("epconfig"))
    assert (name_policy._call_java("endpointName").startswith("ep"))


def test_CustomNamePolicy():
    name_policy = CustomNamePolicy("jobName", "modelname", "epconfig", "ep")
    assert (isinstance(name_policy._to_java(), JavaObject))
    assert (name_policy._call_java("trainingJobName") == "jobName")
    assert (name_policy._call_java("modelName") == "modelname")
    assert (name_policy._call_java("endpointConfigName") == "epconfig")
    assert (name_policy._call_java("endpointName") == "ep")
