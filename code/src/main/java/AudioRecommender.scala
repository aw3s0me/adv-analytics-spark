import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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
    // BROADCAST, to avoid sending sharing data
    // when broadcast we send and store in memory only one copy for each executor (host) in cluster
    // WHY? thousands of tasks that are executed in parallel (in many stages)
    // HOW: 1) cache data as raw java obj 2) cache data across multiple jobs-stages
    val bAliases = spark.sparkContext.broadcast(cleanedData.aliases)

    // create data with counts (TRAIN DATA) and cache it in memory
    // WHY CACHING? USEFUL for ALS (because ALS is iterative)
    val countsTrainData = recommender.buildCounts(rawUserArtistData, bAliases).cache()
    val model = recommender.model(countsTrainData)

    countsTrainData.unpersist()

    // showing feature vector of 10 values
    model.userFactors.select("features").show(truncate = false)
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

  def model(countsTrainData: DataFrame): ALSModel = {
    new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      // contains feature vector of 10 values for each user and product in the model
      // => large user-feature and product-feature matrices
      setRank(10).
      setRegParam(0.01).
      setAlpha(1.0).
      setMaxIter(5).
      setUserCol("user").
      setItemCol("artist").
      setRatingCol("count").
      setPredictionCol("prediction").
      fit(countsTrainData)
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
}