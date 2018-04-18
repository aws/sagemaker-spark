import os
import pytest

from pyspark import SparkConf, SparkContext

from sagemaker_pyspark import classpath_jars
from sagemaker_pyspark.S3Resources import S3DataPath


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


def test_s3_data_path():
    bucket = "bucket"
    prefix = "dir/file"
    s3_obj = S3DataPath(bucket, prefix)

    assert s3_obj.toS3UriString() == "s3://{}/{}".format(bucket, prefix)
