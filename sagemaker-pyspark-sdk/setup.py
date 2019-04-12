#!/usr/bin/env python
from __future__ import print_function
import os
import shutil
import subprocess
import sys
from setuptools import setup


VERSION_PATH = "VERSION"
TEMP_PATH = "deps"
JARS_TARGET = os.path.join(TEMP_PATH, "jars")

in_sagemaker_sdk = os.path.isfile("../sagemaker-spark-sdk/build.sbt")


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read(VERSION_PATH).strip()


try:  # noqa
    if in_sagemaker_sdk:
        try:
            shutil.copyfile(os.path.join("..", VERSION_PATH), VERSION_PATH)
        except OSError:
            print("Could not copy VERSION file")
            exit(1)

        try:
            os.mkdir(TEMP_PATH)
        except OSError:
            print("Could not create dir {0}".format(TEMP_PATH), file=sys.stderr)
            exit(1)

        p = subprocess.Popen("sbt printClasspath".split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             cwd="../sagemaker-spark-sdk/")

        output, errors = p.communicate()

        classpath = []
        # Java Libraries to include.
        java_libraries = ['aws', 'sagemaker', 'hadoop', 'htrace']
        for line in output.decode('utf-8').splitlines():
            path = str(line.strip())
            if path.endswith(".jar") and os.path.exists(path):
                jar = os.path.basename(path).lower()
                if any(lib in jar for lib in java_libraries):
                    classpath.append(path)

        os.mkdir(JARS_TARGET)
        for jar in classpath:
            target_path = os.path.join(JARS_TARGET, os.path.basename(jar))
            if not os.path.exists(target_path):
                shutil.copy(jar, target_path)

        if len(classpath) == 0:
            print("Failed to retrieve the jar classpath. Can't package")
            exit(-1)

    else:
        if not os.path.exists(JARS_TARGET):
            print("You need to be in the sagemaker-pyspark-sdk root folder to package",
                  file=sys.stderr)
            exit(-1)

    setup(
        name="sagemaker_pyspark",
        version=read_version(),
        description="Amazon SageMaker PySpark Bindings",
        author="Amazon Web Services",
        url="https://github.com/aws/sagemaker-spark",
        license="Apache License 2.0",
        zip_safe=False,

        packages=["sagemaker_pyspark",
                  "sagemaker_pyspark.algorithms",
                  "sagemaker_pyspark.transformation",
                  "sagemaker_pyspark.transformation.deserializers",
                  "sagemaker_pyspark.transformation.serializers",
                  "sagemaker_pyspark.jars",
                  "sagemaker_pyspark.licenses"],

        package_dir={
            "sagemaker_pyspark": "src/sagemaker_pyspark",
            "sagemaker_pyspark.jars": "deps/jars",
            "sagemaker_pyspark.licenses": "licenses"
        },
        include_package_data=True,

        package_data={
            "sagemaker_pyspark.jars": ["*.jar"],
            "sagemaker_pyspark.licenses": ["*.txt"]
        },

        scripts=["bin/sagemakerpyspark-jars", "bin/sagemakerpyspark-emr-jars"],

        install_requires=[
            "pyspark>=2.3.2",
            "numpy",
        ],

        setup_requires=["pyspark>=2.3.2", "pypandoc", "pytest-runner", "numpy"],
        tests_require=["pytest", "pytest-cov", "pytest-xdist", "coverage"]
    )

finally:
    if in_sagemaker_sdk:
        if os.path.exists(JARS_TARGET):
            shutil.rmtree(JARS_TARGET)

        if os.path.exists(TEMP_PATH):
            os.rmdir(TEMP_PATH)

        if os.path.exists(VERSION_PATH):
            os.remove(VERSION_PATH)
