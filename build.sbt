logLevel := Level.Error

lazy val root = (project in file("."))
	.enablePlugins(BuildInfoPlugin)
	.settings(
		name := "RegressionModels",
		version := "1.5.3",
		scalaVersion := "2.11.8",
		buildInfoOptions += BuildInfoOption.BuildTime,
		buildInfoPackage := "info",
		ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = false)},
		mainClass in (Compile, run) := Some("RegressionModels")
	)

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "2.1.0",
	"org.apache.spark" %% "spark-sql" % "2.1.0",
	"org.apache.spark" %% "spark-mllib" % "2.1.0",
	"ml.combust.mleap" %% "mleap-runtime" % "0.7.0",
	"ml.combust.mleap" %% "mleap-spark" % "0.7.0",
	"com.typesafe" % "config" % "1.3.1",
	"org.rogach" %% "scallop" % "2.0.1"
)

assemblyMergeStrategy in assembly := {
	case "reference.conf" => MergeStrategy.concat
	case PathList("META-INF", xs @ _*) => MergeStrategy.discard
	case x => MergeStrategy.first
}

assemblyShadeRules in assembly := Seq(
	ShadeRule.rename("com.google.**" -> "shadeio.@1").inAll
)