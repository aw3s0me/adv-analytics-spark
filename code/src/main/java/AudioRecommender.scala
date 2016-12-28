import org.apache.spark.sql.SparkSession

/**
  * Created by akorovin on 28.12.2016.
  */
object AudioRecommender {
  def main(args: Array[String]): Unit = {
    val base = "hdfs:///user/ds/"
    val spark = SparkSession.builder().getOrCreate()

  }
}
