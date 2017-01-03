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
    tree.randomClassifier(trainData, testData)
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

  def classProbabilities(data: DataFrame): Array[Double] = {
    // all number of elements in dataset
    val total = data.count()
    // count number of occurences of distinct categories.
    // (e.g. category1: 20% of dataset, category2: 10% of dataset etc...)
    data.groupBy("Cover_Type").count().
      // order counts by category
      orderBy("Cover_Type").
      select("count").as[Double].
      // to get probability we just divide num of occurences to total number
      map(_ / total).
      collect()
  }

  /**
    * Current accuracy from simple decision tree is ok, but
    * it is not clear is it good or bad
    * => Randomly guessing a classification for each example
    * would produce correct answer
    * ----
    * Pick class at random in proportion its prevalence in training set
    * @param trainData
    * @param testData
    */
  def randomClassifier(trainData: DataFrame, testData: DataFrame): Unit = {
    // find probabilities for training dataset
    val trainPriorProbabilities = classProbabilities(trainData)
    // find probabilities for CV dataset
    val testPriorProbabilities = classProbabilities(testData)
    // calculate accuracy by finding a product of probabilities from training and CV datasets
    // AND SUM UP such probabilities
    val accuracy = trainPriorProbabilities.zip(testPriorProbabilities).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum
    println(accuracy)
  }

  /**
    * Choosing hyperparameters
    * @param trainData
    * @param testData
    */
  def evaluate(trainData: DataFrame, testData: DataFrame): Unit = {
    val assembler = new VectorAssembler().
      setInputCols(trainData.columns.filter(_ != "Cover_Type")).
      setOutputCol("featureVector")

    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val paramGrid = new ParamGridBuilder().
      // GINI - it is the probability that a randomly chosen classification of randomly chosen example is incorrect
      // depends on proportion of examples of class i in subset

      // NOTE - subset that contains only 1 class. has 0 entropy and gini = 0
      // NOTE - low impurity is GOOD
      addGrid(classifier.impurity, Seq("gini", "entropy")).
      // maxDepth - simply limits the number of levels in decision tree
      // NOTE: used to avoid overfitting
      addGrid(classifier.maxDepth, Seq(1, 20)).
      // Disadv: larger number of beans requires more processing time
      // Adv: but may lead to more optimal solution
      addGrid(classifier.maxBins, Seq(40, 300)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()

    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    //spark.sparkContext.setLogLevel("DEBUG")
    val validatorModel = validator.fit(trainData)
    /*
    DEBUG TrainValidationSplit: Got metric 0.6315930234779452 for model trained with {
      dtc_ca0f064d06dd-impurity: gini,
      dtc_ca0f064d06dd-maxBins: 10,
      dtc_ca0f064d06dd-maxDepth: 1,
      dtc_ca0f064d06dd-minInfoGain: 0.0
    }.
    */
    //spark.sparkContext.setLogLevel("WARN")

    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    println(validatorModel.validationMetrics.max)

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData))
    println(testAccuracy)

    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(trainData))
    println(trainAccuracy)
  }

  /**
    * To undo one-hot encoding
    * @param data
    * @return
    */
  def unencodeOneHot(data: DataFrame): DataFrame = {
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    val wildernessAssembler = new VectorAssembler().
      setInputCols(wildernessCols).
      setOutputCol("wilderness")

    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    val withWilderness = wildernessAssembler.transform(data).
      drop(wildernessCols:_*).
      withColumn("wilderness", unhotUDF($"wilderness"))

    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    val soilAssembler = new VectorAssembler().
      setInputCols(soilCols).
      setOutputCol("soil")

    soilAssembler.transform(withWilderness).
      drop(soilCols:_*).
      withColumn("soil", unhotUDF($"soil"))
  }

  def evaluateCategorical(trainData: DataFrame, testData: DataFrame): Unit = {
    // transform one-hot encoded features back to categorical (from numerical binary to categorical)
    val unencTrainData = unencodeOneHot(trainData)
    val unencTestData = unencodeOneHot(testData)

    val assembler = new VectorAssembler().
      setInputCols(unencTrainData.columns.filter(_ != "Cover_Type")).
      setOutputCol("featureVector")

    val indexer = new VectorIndexer().
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini", "entropy")).
      addGrid(classifier.maxDepth, Seq(1, 20)).
      addGrid(classifier.maxBins, Seq(40, 300)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()

    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
    println(testAccuracy)
  }
}