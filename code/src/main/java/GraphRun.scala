import edu.umd.cloud9.collection.XMLInputFormat

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.conf.Configuration

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession, Row}
import org.apache.spark.sql.functions._

import scala.xml._

/**
  * Created by akorovin on 04.01.2017.
  * Goal of this chapter - use GraphX to acquire, transform
  * network of MeSH terms (Semantic tags - Medical Subject Headings)
  * using MEDLINE data (apply MeSH on MEDLINE)
  */
object GraphRun {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val medlineRaw: Dataset[String] = loadMedline(spark, "hdfs:///user/ds/medline")
    val medline: Dataset[Seq[String]] = medlineRaw.map(majorTopics).cache()
  }

  /**
    * Fetching only major articles (getting major tags)
    * @param record
    * @return
    */
  def majorTopics(record: String): Seq[String] = {
    // elem is instance of Elem class (scala represents it as individual Node)
    val elem = XML.loadString(record)
    // operator '\\' - RETRIEVING NON-DIRECT CHILDREN
    val dn = elem \\ "DescriptorName"
    // 1. operator '\' - RETRIEVING DIRECT CHILDREN
    // NOTE: ONLY WORKS WITH DIRECT CHILDREN
    // 2. call text function to retrieve MeSH tags within each node
    // 3. filter entries that are major (MajorTopicYN == 'Y')
    // NOTE: need to preface attribute name with @ symbol (MajorTopicYN - is attribute)
    val mt = dn.filter(n => (n \ "@MajorTopicYN").text == "Y")
    mt.map(n => n.text)
  }

  /**
    * load MEDLINE XML database into shell
    * (academic papers published in life science/medicine journals)
    * @param spark
    * @param path
    * @return
    */
  def loadMedline(spark: SparkSession, path: String): Dataset[String] = {
    import spark.implicits._
    val conf = new Configuration()
    // each entry is MedlineCitation record (info about publication)
    // set start tag configuration parameter to be prefix of the MedlineCitation start tag
    // (because values can change from record to record)
    conf.set(XMLInputFormat.START_TAG_KEY, "<MedlineCitation ")
    conf.set(XMLInputFormat.END_TAG_KEY, "</MedlineCitation>")
    val sc = spark.sparkContext
    // XmlInputFormat will return these varying attributes in the record values
    val in = sc.newAPIHadoopFile(path, classOf[XMLInputFormat],
      classOf[LongWritable], classOf[Text], conf)
    in.map(line => line._2.toString).toDS()
  }
}

class GraphRun(private val spark: SparkSession) {

}
