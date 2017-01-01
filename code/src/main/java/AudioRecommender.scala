import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by akorovin on 28.12.2016.
  */
object AudioRecommender {
  def main(args: Array[String]): Unit = {
    // assuming that files are available under /user/ds/
    // full path is located in core-site.xml (property fs.defaultFS)
    val base = "hdfs://localhost:19000/user/ds/"
    val spark = SparkSession
      .builder()
      .master("local")
      .getOrCreate()

    // format (divided by space): userID artistID playCount
    val rawUserArtistData = spark.read.textFile(base + "user_artist_data.txt")
    // Artist data. format: artistId artistName
    val rawArtistData = spark.read.textFile(base + "artist_data.txt")
    // maps artist ids that may be misspelled to their canonical names
    // format: goodId badId
    val rawArtistAlias = spark.read.textFile(base + "artist_alias.txt")

    val recommender = new AudioRecommender(spark)
    val cleanedData = recommender.preparation(rawUserArtistData, rawArtistData, rawArtistAlias)
    println(cleanedData.userArtistDF.schema)
  }
}

case class RecommenderDataCleaned(userArtistDF: DataFrame,
                                  artistDF: DataFrame,
                                  aliases: Map[Int, Int])

class AudioRecommender(private val spark: SparkSession) {
  // without import gives exception
  import spark.implicits._

  /**
    * Clean data and create needed dataframes and alias map
    * @param rawUserArtistData
    * @param rawArtistData
    * @param rawArtistAlias
    * @return
    */
  def preparation(rawUserArtistData: Dataset[String],
                  rawArtistData: Dataset[String],
                  rawArtistAlias: Dataset[String]): RecommenderDataCleaned = {

    // convert dataset of strings to dataframe
    val userArtistDF = this.buildUserArtistData(rawUserArtistData)
    val artistDF = this.buildArtistData(rawArtistData)
    val artistAlias = this.buildArtistAlias(rawArtistAlias)

    RecommenderDataCleaned(userArtistDF, artistDF, artistAlias)
  }

  def model(
             rawUserArtistData: Dataset[String],
             rawArtistData: Dataset[String],
             rawArtistAlias: Dataset[String]): Unit = {
    // BROADCAST, to avoid sending sharing data
    // when broadcast we send and store in memory only one copy for each executor (host) in cluster
    // WHY? thousands of tasks that are executed in parallel (in many stages)
    // HOW: 1) cache data as raw java obj 2) cache data across multiple jobs-stages
    val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))

    // create data with counts (TRAIN DATA) and cache it in memory
    // WHY CACHING? USEFUL for ALS (because ALS is iterative)
    val trainData = buildCounts(rawUserArtistData, bArtistAlias).cache()

    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      // rank - number of latent factors in the model and equivalently
      // number of columns k in the user-feature and product-feature matrices
      setRank(10).
      // lambda - standard overfitting parameter
      // Higher values resist overfitting, but values that are too high hurt the factorization’s accuracy.
      setRegParam(0.01).
      // alpha - Controls the relative weight of observed versus unobserved user-product interactions in the factorization.
      setAlpha(1.0).
      // iterations - number of iterations that the factorization runs
      // more - better factorization
      setMaxIter(5).
      setUserCol("user").
      setItemCol("artist").
      setRatingCol("count").
      setPredictionCol("prediction").
      fit(trainData)

    trainData.unpersist()
    // showing feature vector of 10 values
    model.userFactors.select("features").show(truncate = false)
    // examine a user and recommendations for him
    val userID = 2093760
    // extract ids of artists that this user listened
    val existingArtistIDs = trainData.
      filter($"user" === userID).
      select("artist").as[Int].collect()

    val artistByID = buildArtistData(rawArtistData)

    // show artists by id
    /**
      * +-------+---------------+
      * |     id|           name|
      * +-------+---------------+
      * |   1180|     David Gray|
      * |    378|  Blackalicious|
        |    813|     Jurassic 5|
        |1255340|The Saw Doctors|
        |    942|         Xzibit|
        +-------+---------------+
      */
    artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

    val topRecommendations = makeRecommendations(model, userID, 5)
    /**
      * Results of making 5 recommendations by user ID
      * NOTE: values are between 0 and 1
      * BIGGER VALUE = BETTER RECOMMENDATION
      * NOTE: IT IS NOT A PROBABILITY
      * +-------+-----------+
      * | artist| prediction|
      * +-------+-----------+
      * |1300642|0.030599192|
      * |   2814|0.030095732|
      * |1001819|0.029307712|
      * |1037970| 0.02915808|
      * |1007614|0.028896002|
      * +-------+-----------+
      */
    topRecommendations.show()

    val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()

    /**
      * +-------+----------+
      * |     id|      name|
      * +-------+----------+
      * |   2814|   50 Cent|
      * |1007614|     Jay-Z|
      * |1037970|Kanye West|
      * |1001819|      2Pac|
      * |1300642|  The Game|
      * +-------+----------+
      */
    artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

    model.userFactors.unpersist()
    model.itemFactors.unpersist()
  }

  def evaluate(countsTrainData: DataFrame): Unit = {
    // split data into training and CV(CROSS-VALIDATION) set
    // TRAINING of MODEL - train data (90%), VALIDATION (EVALUATION) - CV data (10%)
    val Array(trainData, cvData) = countsTrainData.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    cvData.cache()

    // remove duplicates and collect to driver
    val allArtistIDs = countsTrainData.select("artist").as[Int].distinct().collect()
    // broadcast collected artists ids
    val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)
    // calculate AUC on cross-validation data
    val mostListenedAUC = areaUnderCurve(cvData, bAllArtistIDs, predictMostListened(trainData))
    println(mostListenedAUC)

    // CHOOSING HYPERPARAMETERS. BRUTE FORCE
    val evaluations =
      for (rank     <- Seq(5,  30);
           regParam <- Seq(1.0, 0.0001);
           alpha    <- Seq(1.0, 40.0))
        yield {
          val model = new ALS().
            setSeed(Random.nextLong()).
            setImplicitPrefs(true).
            setRank(rank).setRegParam(regParam).
            setAlpha(alpha).setMaxIter(20).
            setUserCol("user").setItemCol("artist").
            setRatingCol("count").setPredictionCol("prediction").
            fit(trainData)

          val auc = areaUnderCurve(cvData, bAllArtistIDs, model.transform)

          model.userFactors.unpersist()
          model.itemFactors.unpersist()

          (auc, (rank, regParam, alpha))
        }
    // sort descending
    // conclusion: alpha = 40 is better (it is proposed as default in paper)
    // higher lambda looks slightly better
    // rank doesnt make clear difference. 50 in both highest and lowest combinations
    evaluations.sorted.reverse.foreach(println)

    trainData.unpersist()
    cvData.unpersist()
  }

  def buildUserArtistData(rawUserArtistData: Dataset[String]): DataFrame = {
    rawUserArtistData.map { line =>
      val Array(user, artist, _*) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")
  }

  def buildArtistData(rawArtistData: Dataset[String]): DataFrame = {
    // CAN'T USE MAP, BECAUSE SOME VALUES (artist names) ARE 0 and we need to avoid them
    // IDEA of MAP - one input one output => need to use flatMap
    rawArtistData.flatMap { line =>
      // SPAN - splits line by its first tab by consuming characters that are not tabs
      // return tuple and then unpack it to id and name
      val (id, name) = line.span(_ != '\t')
      // NOTE: SOME NAMES ARE EMPTY. => cause NumberFormatException (need to deal with it)
      if (name.isEmpty) {
        // NOTE: None and Some are used in flat map to show if value exists or not
        None
      }

      try {
        // then parse numeric id and remove tabs and whitespaces in name
        Some((id.toInt, name.trim))
      } catch {
        // if we couldnt parse id => return nothing (value doesnt exist)
        case _: NumberFormatException => None
      }
    }.toDF("id", "name")
  }

  /**
    * Maps "BAD" artist names to "GOOD"
    * @param rawArtistAlias
    * @return
    */
  def buildArtistAlias(rawArtistAlias: Dataset[String]): Map[Int,Int] = {
    rawArtistAlias.flatMap { line =>
      // artist - goodName, alias - badName
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap
  }

  def buildCounts(rawUserArtistData: Dataset[String],
                  artistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
    rawUserArtistData.map {line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      // get artist alias if exists else get original artist
      val finalArtistID = artistAlias.value.getOrElse(artistID, artistID)
      (userID, finalArtistID, count)
    }.toDF("user", "artist", "count")
  }

  def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
    val toRecommend = model.itemFactors.
      select($"id".as("artist")).
      withColumn("user", lit(userID))
    model.transform(toRecommend).
      select("artist", "prediction").
      orderBy($"prediction".desc).
      limit(howMany)
  }

  /**
    * Accepts CV(CROSS-VALIDATION) set as positive(good) artist for each user
    * And prediction function that TRANSLATES each
    * user-artist as a PREDICTION. format - Dataset("user", "artist", "rating-value")
    * @param positiveData
    * @param bAllArtistIDs
    * @param predictFunction
    * @return
    */
  def areaUnderCurve(
                      positiveData: DataFrame,
                      bAllArtistIDs: Broadcast[Array[Int]],
                      predictFunction: (DataFrame => DataFrame)): Double = {

    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive".
    // Make predictions for each of them, including a numeric score
    val positivePredictions = predictFunction(positiveData.select("user", "artist")).
      withColumnRenamed("prediction", "positivePrediction")

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other artists, excluding those that are "positive" for the user.
    val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
      groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0
        // Make at most one pass over all artists to avoid an infinite loop.
        // Also stop when number of negative equals positive set size
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
          // Only add new distinct IDs
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // Return the set with user ID added back
        negative.map(artistID => (userID, artistID))
      }.toDF("user", "artist")

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user", "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user").as("correct")).
      select("user", "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, "user").
      select($"user", ($"correct" / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }

  def predictMostListened(train: DataFrame)(allData: DataFrame): DataFrame = {
    val listenCounts = train.groupBy("artist").
      agg(sum("count").as("prediction")).
      select("artist", "prediction")
    allData.
      join(listenCounts, Seq("artist"), "left_outer").
      select("user", "artist", "prediction")
  }
}