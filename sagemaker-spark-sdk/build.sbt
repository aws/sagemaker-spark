organization := "com.amazonaws"
organizationName := "Amazon Web Services"
organizationHomepage := Some(url("https://aws.amazon.com"))
name := "sagemaker-spark"
description := "SageMaker Spark provides a Spark interface to Amazon SageMaker, allowing " +
  "customers to train using the Spark Estimator API, host their model on Amazon SageMaker, and " +
  "make predictions with their model using the Spark Transformer API."
homepage := Some(url("https://github.com/aws/sagemaker-spark"))
scmInfo := Some(
  ScmInfo(
    url("https://github.com/aws/sagemaker-spark"),
    "https://github.com/aws/sagemaker-spark.git"
  )
)
licenses := Seq("Apache License, Version 2.0" -> url("https://aws.amazon.com/apache2.0"))

scalaVersion := "2.11.7"

// to change the version of spark add -DSPARK_VERSION=2.x.x when running sbt
// for example: "sbt -DSPARK_VERSION=2.1.1 clean compile test doc package"
val sparkVersion = System.getProperty("SPARK_VERSION", "2.2.0")
version := "spark_" + sparkVersion + "-1.1.1"

lazy val SageMakerSpark = (project in file("."))

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-aws" % "2.8.1",
  "com.amazonaws" % "aws-java-sdk-s3" % "1.11.238",
  "com.amazonaws" % "aws-java-sdk-sts" % "1.11.238",
  "com.amazonaws" % "aws-java-sdk-sagemaker" % "1.11.238",
  "com.amazonaws" % "aws-java-sdk-sagemakerruntime" % "1.11.238",
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.scalatest" %% "scalatest" % "3.0.4" % "test",
  "org.mockito" % "mockito-all" % "1.10.19" % "test"
)

// add a task to print the classpath. Also use the packaged JAR instead
// of the .class files.
exportJars := true
lazy val printClasspath = taskKey[Unit]("Dump classpath")
printClasspath := (fullClasspath in Runtime value) foreach { e => println(e.data) }

// set coverage threshold
coverageMinimum := 90
coverageFailOnMinimum := true

// make scalastyle gate the build
(compile in Compile) := {
  scalastyle.in(Compile).toTask("").value
  (compile in Compile).value
}

(test in Test) := {
  scalastyle.in(Test).toTask("").value
  (test in Test).value
}

parallelExecution in Test := false

// publishing configuration
publishMavenStyle := true
pomIncludeRepository := { _ => false }
publishArtifact in Test := false
publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}
pomExtra := (
  <developers>
    <developer>
      <id>amazonwebservices</id>
      <organization>Amazon Web Services</organization>
      <organizationUrl>https://aws.amazon.com</organizationUrl>
      <roles>
        <role>developer</role>
      </roles>
    </developer>
  </developers>
  )
