import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import scala.util.Random
/**
  * Created by akorovin on 01.01.2017.
  */
object DecisionTree {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val dataWithoutHeader = spark.read.
      option("inferSchema", value = true).
      option("header", value = false).
      csv("hdfs:///user/ds/covtype.data")

    val colNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++
      // one-hot encoding of categorical type with 4 values
      (0 until 4).map(i => s"Wilderness_Area_$i") ++
      // one-hot encoding of categorical type with 40 values
      (0 until 40).map(i => s"Soil_Type_$i") ++
      Seq("Cover_Type")

    val data = dataWithoutHeader.toDF(colNames:_*).
      withColumn("Cover_Type", $"Cover_Type".cast("double"))

    data.show()
    data.head

    // Split into 90% train (+ CV), 10% test
    val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    testData.cache()

    val tree = new DecisionTree(spark)

    tree.simpleDecisionTree(trainData, testData)
  }
}

class DecisionTree(private val spark: SparkSession) {
  import spark.implicits._

  def simpleDecisionTree(trainData: DataFrame, testData: DataFrame): Unit = {

    val assembler = new VectorAssembler().
      setInputCols(trainData.columns.filter(_ != "Cover_Type")).
      setOutputCol("featureVector")

    val assembledTrainData = assembler.transform(trainData)
    assembledTrainData.select("featureVector").show(truncate = false)

    // DecisionTreeClassifier is used suggests that target value is distinct category number
    // not a numeric value => USED TO SEPARATE DATASET INTO CLASSES (LABELS)
    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    // TRAIN Decision trees model
    val model = classifier.fit(assembledTrainData)
    println(model.toDebugString)

    model.featureImportances.toArray.zip(trainData.columns).
      sorted.reverse.foreach(println)

    val predictions = model.transform(assembledTrainData)

    predictions.select("Cover_Type", "prediction", "probability").
      show(truncate = false)

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction")
    // summarize accuracy with a single number
    /**
      * Computing precision for each category:
      * (0 until 7).map(
          cat => (metrics.precision(cat), metrics.recall(cat))
        ).foreach(println)
      */
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println(accuracy)
    println(f1)
    // get predictions
    val predictionRDD = predictions.
      select("prediction", "Cover_Type").
      as[(Double,Double)].rdd
    // computes metrics that measure the quality of predictions
    // NOTE: use Binary Classification Metrics for binary classifier
    val multiclassMetrics = new MulticlassMetrics(predictionRDD)
    println(multiclassMetrics.confusionMatrix)
    // helpful to look into confusion matrix. 7x7 size
    // each row corresponds to an actual correct value,
    // and each column to a predicted value
    // ------
    // The entry at row i and column j counts the number of times
    // an example with true category i was predicted as category j.
    // ------
    // the correct predictions are the counts along the diagonal, and incorrect predictions are everything else
    // ------
    // GOOD when: Counts are high along the diagonal, which is good
    val confusionMatrix = predictions.
      groupBy("Cover_Type").
      // number of values we want to encounter
      pivot("prediction", (1 to 7)).
      count().
      na.fill(0.0).
      orderBy("Cover_Type")

    confusionMatrix.show()
  }
}