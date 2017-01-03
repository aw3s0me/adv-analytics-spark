import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, StringIndexer, StandardScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Random

/**
  * Created by akorovin on 03.01.2017.
  */
object KMeans {
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

    val runKMeans = new KMeans(spark)
  }
}

/**
  * Building a system to detect anomalous network traffic
  * @param spark
  */
class KMeans(private val spark: SparkSession) {
  import spark.implicits._

}
