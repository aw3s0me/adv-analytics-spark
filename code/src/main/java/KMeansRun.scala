import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, StringIndexer, StandardScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Random

/**
  * Created by akorovin on 03.01.2017.
  */
object KMeansRun {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      // kddcup 99 data. Network intrusion.
      csv("hdfs:///user/ds/kddcup.data").
      // data contains 38 features
      // each connection - is one line of CSV-file
      toDF(
        // some features are non-numeric, like protocol type (http, tcp, udp).
        // BUT KMEANS REQUIRES NUMBERS
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        // some features indicate presence or absence of behavior
        // like su_attempted
        // they look like one-hot encoded categorical features (but not grouped)
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        // rest are ratios. taking values from 0.0 to 1.0
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        // NOTE: label is given in the last field. most are labelled 'normal'
        "label")

    data.cache()

    val runKMeans = new KMeansRun(spark)
  }
}

/**
  * Building a system to detect anomalous network traffic
  * @param spark
  */
class KMeansRun(private val spark: SparkSession) {
  import spark.implicits._

  // Clustering, Take 0. Fitting without setting up number of clusters
  def clusteringTake0(data: DataFrame): Unit = {
    // exploring data set
    // which labels present in the data and how many are there of each
    // --- counts by label and sorts them by descending order (23 distinct labels)
    // CONCLUSION => we need k = 23 clusters (because we have 23 different cases)
    data.select("label").groupBy("label").count().orderBy($"count".desc).show(25)

    // remove the three categorical value columns.
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    // remaining are converted to a vector of features
    val assembler = new VectorAssembler().
      setInputCols(numericOnly.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    // create KMeans model
    // kmeans operates only on features (need to set features col)
    // CURRENTLY we do no set number of clusters
    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
    val pipelineModel = pipeline.fit(numericOnly)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // print out centroids. will print 2 centroids (k=2)
    kmeansModel.clusterCenters.foreach(println)

    val withCluster = pipelineModel.transform(numericOnly)
    // count labels within each cluster
    withCluster.select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy($"cluster", $"count".desc).
      show(25)
    // will print out that only one data point ended in cluster 1
    // => clustering with k = 2 is unsufficient
    numericOnly.unpersist()
  }

  // Clustering, Take 1. fitting with setting up the number of centres
  // but without maximum number of iterations
  def clusteringScore0(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  /**
    * Setup also other parameters to compare with fitting
    * @param data
    * @param k
    * @return
    */
  def clusteringScore1(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  def clusteringTake1(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    // (x to y by z) scala way to create collection between start and end with given step
    (20 to 100 by 20).map(k => (k, clusteringScore0(numericOnly, k))).foreach(println)
    (20 to 100 by 20).map(k => (k, clusteringScore1(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  // Clustering, Take 2. Feature normalization
  def clusteringScore2(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    // trying to normalize feature (subract mean from each feature and divide by std)
    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      // however subtracting mean has no effect on clustering
      // because shifting was by the same amount in the same directions
      // => Doesnt affect interpoint euclidean distances
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
  }

  def clusteringTake2(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    (60 to 270 by 30).map(k => (k, clusteringScore2(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  // Clustering, Take 3. Dealing with categorical features
  /**
    * Function to convert categorical features (protocol service flag) to numerical features
    * Using one-hot encoding
    * @param inputCol
    * @return
    */
  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")
    // use predefined class. OneHotEncoder
    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, inputCol + "_vec")
  }

  def clusteringScore3(data: DataFrame, k: Int): Double = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
  }

  def clusteringTake3(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore3(data, k))).foreach(println)
  }

  // Clustering, Take 4. Calculate cluster score
  def entropy(counts: Iterable[Int]): Double = {
    val values = counts.filter(_ > 0)
    val n = values.map(_.toDouble).sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }

  def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    pipeline.fit(data)
  }

  def clusteringScore4(data: DataFrame, k: Int): Double = {
    val pipelineModel = fitPipeline4(data, k)

    // Predict cluster for each datum
    val clusterLabel = pipelineModel.transform(data).
      select("cluster", "label").as[(Int, String)]
    val weightedClusterEntropy = clusterLabel.
      // Extract collections of labels, per cluster
      groupByKey { case (cluster, _) => cluster }.
      mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq
        // Count labels in collections
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)
      }.collect()

    // Average entropy weighted by cluster size
    weightedClusterEntropy.sum / data.count()
  }

  def clusteringTake4(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore4(data, k))).foreach(println)

    val pipelineModel = fitPipeline4(data, 180)
    val countByClusterLabel = pipelineModel.transform(data).
      select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy("cluster", "label")
    countByClusterLabel.show()
  }

  // Detect anomalies. Final anomaly detector.
  def buildAnomalyDetector(data: DataFrame): Unit = {
    val pipelineModel = fitPipeline4(data, 180)

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters

    val clustered = pipelineModel.transform(data)
    // choose threshold 100-th farthest data point from among known data
    val threshold = clustered.
      select("cluster", "scaledFeatureVector").as[(Int, Vector)].
      map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
      orderBy($"value".desc).take(100).last

    val originalCols = data.columns
    // takes new data point. if new data points distance exceeds threshold
    // => IT IS ANOMALOUS
    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledFeatureVector")
      // apply this threshold to all new arriving data
      Vectors.sqdist(centroids(cluster), vec) >= threshold
    }.select(originalCols.head, originalCols.tail:_*)

    println(anomalies.first())
  }
}
