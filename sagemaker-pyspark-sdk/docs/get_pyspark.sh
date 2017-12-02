#!/bin/bash


curl -LO https://github.com/apache/spark/archive/branch-2.2.zip
unzip -o branch-2.2.zip "spark-branch-2.2/python/pyspark/*"
cp -r spark-branch-2.2/python/pyspark ../src/
