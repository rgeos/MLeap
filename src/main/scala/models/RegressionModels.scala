package models

import ml.combust.bundle.{BundleFile => BF}
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.spark.SparkSupport._
import resource._
import org.apache.log4j.{Level => l, Logger => L}
import org.apache.spark.sql.{DataFrame, SparkSession => SS}
import org.apache.spark.ml.{PipelineStage, Pipeline => P, PipelineModel => PM}
import org.apache.spark.ml.bundle.{SparkBundleContext => SBC}
import org.apache.spark.ml.feature.{VectorAssembler => VA, StandardScaler => SSc}
import org.apache.spark.ml.regression.{LinearRegression => LR, RandomForestRegressor => RF}


object RegressionModels {

	L.getLogger("org").setLevel(l.ERROR)

	def main(args: Array[String]): Unit = {

		val spark: SS = SS.builder().master("local").appName("LR").getOrCreate()
		if(args.length < 3) System.err.println("You need at least three arguments")

		val Array(path: String, file: String, format: String) = args
		val startTime = System.nanoTime()
		start(path, file, format, spark)

		val elapsedTime = (System.nanoTime() - startTime) / 1e9
		println(s"Total elapsed time $elapsedTime")
	}


	def readFile(path: String, file: String, format: String = "csv", spark: SS): DataFrame =
		spark.read
			.options(Map("header" -> "true", "inferSchema" -> "true"))
			.format(format)
			.load(path + "/" + file)


	def export_pipeline(pipeline: PM, path: String = "/tmp/model_.zip"): Unit = {

		val sbc = SBC()
		for(bf <- managed(BF(s"jar:file:$path"))) {
			pipeline.writeBundle.format(SerializationFormat.Json).save(bf)(sbc).get
		}
	}


	def start(path: String, file: String, format: String = "csv", spark: SS): Unit = {

		val data = readFile(path, file, format, spark)
		data.cache()

//		data.printSchema()
//		data.describe().show()

		val features = Array("temperature", "exhaust_vacuum", "ambient_pressure", "relative_humidity")
		val allCols = features.union(Seq("energy_output")).map(data.col)

		val nullFilter = allCols.map(_.isNotNull).reduce(_ && _)

		val dataFiltered = data.select(allCols: _*).filter(nullFilter).persist()

//		dataFiltered.describe().show()

		// split the dataset
//		val Array(training, test) = dataFiltered.randomSplit(Array(.7, .3), seed = 196)

		// feature pipeline
		val featureAssembler	= new VA(uid = "feature_assembler").setInputCols(features).setOutputCol("unscaled_features")
		val featureScaler			= new SSc(uid = "feature_scaler").setInputCol("unscaled_features").setOutputCol("scaled_features")

		// assemble the pipeline
		val estimators: Array[PipelineStage] = Array(featureAssembler, featureScaler)

		val featurePipeline = new P(uid = "feature_pipeline").setStages(estimators)

		val sparkFeaturePipelineModel = featurePipeline.fit(dataFiltered)

		println("Done with the pipeline")

		// train a random forest
		val randomForest = new RF(uid = "random_forest_regression").
			setFeaturesCol("scaled_features").
			setLabelCol("energy_output").
			setPredictionCol("energy_prediction")

		val randomForestModel = new P().setStages(Array(sparkFeaturePipelineModel, randomForest)).fit(dataFiltered)

		println("Completed Random Forest Training")

		// train a linear regression model
		val linearRegression = new LR(uid = "linear_regression").
			setFeaturesCol("scaled_features").
			setLabelCol("energy_output").
			setPredictionCol("energy_prediction")

		val linearRegressionModel = new P().setStages(Array(sparkFeaturePipelineModel, linearRegression)).fit(dataFiltered)

		println("Completed Linear Regression Training")

		// export the pipelines
		export_pipeline(randomForestModel, "/tmp/models/randomForest.zip")
		export_pipeline(linearRegressionModel, "/tmp/models/linearRegression.zip")

		spark.stop()

	}
}


