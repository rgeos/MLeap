// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC # Linear Regression using ![LOGO](http://demo.epigno.systems/scala_spark.png)
// MAGIC 
// MAGIC In this notebook we will will employ a simple linear regression model to predict the amount of energy output of a power plant. The dataset used for this analysis comes from [UC Irvine machine learning repository](http://mlr.cs.umass.edu/ml/datasets/Combined+Cycle+Power+Plant). The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011).
// MAGIC 
// MAGIC **Data information** as described on the site above:
// MAGIC 
// MAGIC Features consist of hourly average ambient variables 
// MAGIC - Temperature (T) in the range 1.81C and 37.11C,
// MAGIC - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
// MAGIC - Relative Humidity (RH) in the range 25.56% to 100.16%
// MAGIC - Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
// MAGIC - Net hourly electrical energy output (EP) 420.26-495.76 MW
// MAGIC The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization.
// MAGIC 
// MAGIC The original headers were renamed as below:
// MAGIC - T  -> temperature
// MAGIC - V  -> exhaust\_vacuum
// MAGIC - AP -> ambient\_pressure
// MAGIC - RH -> relative\_humidity
// MAGIC - EP -> energy\_output
// MAGIC 
// MAGIC Our goal is to predict the `energy_output` (label) based on the other four features.
// MAGIC 
// MAGIC Alternative data [Link](http://www.caiso.com/Pages/TodaysOutlook.aspx#SupplyandDemand)

// COMMAND ----------

// importing the necessary libraries

import org.apache.spark.ml.feature.{VectorAssembler => VA}
import org.apache.spark.ml.regression.{LinearRegression => LR}
import org.apache.spark.sql.{SparkSession => SS}
import org.apache.spark.ml.linalg.Vectors

// COMMAND ----------

// the data
val file_name = "/FileStore/tables/6zm535q61494044083775/data.csv"

// COMMAND ----------

// creating a spark context and loading the data
val spark = SS.builder().getOrCreate()
val file_name = "/FileStore/tables/6zm535q61494044083775/data.csv"
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load(file_name)

// COMMAND ----------

// print the schema of the dataframe
data.cache()
data.printSchema()

// COMMAND ----------

// simple data description
display(data.describe())

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC We will need a transformer to combine all the features into a single vector. That can be achieved in spark using the `VectorAssembler` library. [VectorAssembler APIs](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.VectorAssembler)
// MAGIC 
// MAGIC Here are some examples for Scala, Java & Python. [Details](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)

// COMMAND ----------

// define the features into an array
val features = Array("temperature", "exhaust_vacuum", "ambient_pressure", "relative_humidity")

// COMMAND ----------

// prepare the data
var lr_data = data.select($"energy_output".as("label"), $"temperature", $"exhaust_vacuum", $"ambient_pressure", $"relative_humidity")
lr_data.printSchema()

// COMMAND ----------

// split the dataset into training and test
val Array(training, test) = lr_data.randomSplit(Array(.7, .3), seed = 196)

// COMMAND ----------

// A vector is what the ML algorithm reads to train a model
val training_vector	= new VA().setInputCols(features).setOutputCol("features").transform(training).select($"label", $"features")
val test_vector		= new VA().setInputCols(features).setOutputCol("features").transform(test).select($"label", $"features")

// COMMAND ----------

// Create a Linear Regression Model object
val lr = new LR()

// Fit the model to the data
val model = lr.fit(training_vector)

// We use explain params to dump the parameters we can use
// lr.explainParams()

// COMMAND ----------

// run the model on the test data
val results = model.transform(test_vector)

// results.show()

// COMMAND ----------

// evaluate the model
import org.apache.spark.ml.evaluation.{RegressionEvaluator => RE}

// Root Mean Square Error
val eval = new RE().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
println("RMSE: %.3f".format(eval.evaluate(results)))

// Mean Square Error
eval.setMetricName("mse")
println("MSE: %.3f".format(eval.evaluate(results)))

// Mean Absolute Error
eval.setMetricName("mae")
println("MAE: %.3f".format(eval.evaluate(results)))

// r2
eval.setMetricName("r2")
println("R2: %.3f".format(eval.evaluate(results)))


// COMMAND ----------


