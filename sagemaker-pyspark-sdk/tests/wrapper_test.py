import os
import pytest

from pyspark import SparkConf, SparkContext

from sagemaker_pyspark import (S3DataPath, EndpointCreationPolicy, RandomNamePolicyFactory,
                               SageMakerClients, IAMRole, classpath_jars)
from sagemaker_pyspark.wrapper import Option, ScalaMap, ScalaList


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


def test_convert_dictionary():
    dictionary = {"key": "value"}
    map = ScalaMap(dictionary)._to_java()
    assert getattr(map, "apply")("key") == "value"


def test_convert_list():
    list = ["features", "label", "else"]
    s_list = ScalaList(list)._to_java()
    assert getattr(s_list, "apply")(0) == "features"
    assert getattr(s_list, "apply")(1) == "label"
    assert getattr(s_list, "apply")(2) == "else"


def test_convert_option():
    list = ["features", "label", "else"]
    option = Option(list)._to_java()
    assert getattr(getattr(option, "get")(), "apply")(0) == "features"