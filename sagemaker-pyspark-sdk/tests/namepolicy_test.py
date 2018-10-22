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
    java_obj = CustomNamePolicyFactory("jobName", "modelname", "epconfig", "ep")
    assert(isinstance(java_obj._to_java(), JavaObject))


def test_CustomNamePolicyWithTimeStampSuffixFactory():
    java_obj = CustomNamePolicyWithTimeStampSuffixFactory("jobName", "modelname", "epconfig", "ep")
    assert(isinstance(java_obj._to_java(), JavaObject))


def test_CustomNamePolicyWithTimeStampSuffix():
    java_obj = CustomNamePolicyWithTimeStampSuffix("jobName", "modelname", "epconfig", "ep")
    assert(isinstance(java_obj._to_java(), JavaObject))


def test_CustomNamePolicy():
    java_obj = CustomNamePolicy("jobName", "modelname", "epconfig", "ep")
    assert (isinstance(java_obj._to_java(), JavaObject))
