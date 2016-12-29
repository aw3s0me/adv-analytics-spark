import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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

  def preparation(rawUserArtistData: Dataset[String],
                  rawArtistData: Dataset[String],
                  rawArtistAlias: Dataset[String]): RecommenderDataCleaned = {

    // convert dataset of strings to dataframe
    val userArtistDF = this.buildUserArtistData(rawUserArtistData)
    val artistDF = this.buildArtistData(rawArtistData)
    val artistAlias = this.buildArtistAlias(rawArtistAlias)

    RecommenderDataCleaned(userArtistDF, artistDF, artistAlias)
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
}