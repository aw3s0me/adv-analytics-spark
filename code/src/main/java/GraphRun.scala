import edu.umd.cloud9.collection.XMLInputFormat
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

import com.google.common.hash.Hashing
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.conf.Configuration
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
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
    // MEDLINE data.
    // each entry is citation that contains multiple topics [Citation 1:M Topic]
    val medlineRaw: Dataset[String] = loadMedline(spark, "hdfs:///user/ds/medline")
    val medline: Dataset[Seq[String]] = medlineRaw.map(majorTopics).cache()

    // get topics from tags
    val topics = medline.flatMap(mesh => mesh).toDF("topic")
    topics.createOrReplaceTempView("topics")
    // calculate occurrences of topic (frequency) and order them
    val topicDist = spark.sql("SELECT topic, COUNT(*) cnt FROM topics GROUP BY topic ORDER BY cnt DESC")
    topicDist.show()
    topicDist.createOrReplaceTempView("topic_dist")
    // group by count and take 10 most frequent
    spark.sql("SELECT cnt, COUNT(*) dist FROM topic_dist GROUP BY cnt ORDER BY dist DESC LIMIT 10").show()

    // to create all combinations between medline topics
    /**
      * Usage of combinations:
      * val list = List(1, 2, 3)
        val combs = list.combinations(2)
        combs.foreach(println)
      * // returns List(1,2), List(1,3), List(2,3)
      */
    // NOTE: COMBINATIONS DEPENDS ON LIST ORDER
    // => so need to sort list (use property sorted)
    val topicPairs = medline.flatMap(_.sorted.combinations(2)).toDF("pairs")
    topicPairs.createOrReplaceTempView("topic_pairs")
    // count co-occurrences
    val cooccurs = spark.sql("SELECT pairs, COUNT(*) cnt FROM topic_pairs GROUP BY pairs")
    cooccurs.cache()

    cooccurs.createOrReplaceTempView("cooccurs")
    println("Number of unique co-occurrence pairs: " + cooccurs.count())
    spark.sql("SELECT pairs, cnt FROM cooccurs ORDER BY cnt DESC LIMIT 10").show()

    val vertices = topics.map { case Row(topic: String) => (hashId(topic), topic) }.toDF("hash", "topic")
    val edges = cooccurs.map { case Row(topics: Seq[_], cnt: Long) =>
      // left vertex id must be smaller than right vertex id
      // => we sort hashes
      val ids = topics.map(_.toString).map(hashId).sorted
      Edge(ids(0), ids(1), cnt)
    }
    val vertexRDD = vertices.rdd.map{ case Row(hash: Long, topic: String) => (hash, topic) }
    val topicGraph = Graph(vertexRDD, edges.rdd)
    topicGraph.cache()

    // want to know whether or not it is connected
    // in a connected graph it is possible to reach from one vertex to any other
    // NOTE: if graph is not connected - better to divide it into separate components and treat individually
    val connectedComponentGraph = topicGraph.connectedComponents()
    // Each vertex is a tuple of (vertexId, component id)
    val componentDF = connectedComponentGraph.vertices.toDF("vid", "cid")
    // find a list of all connected components and their sizes (by grouping their component id)
    val componentCounts = componentDF.groupBy("cid").count()
    // if we look. first largest component contains 90% of all vertices
    componentCounts.orderBy(desc("count")).show()

    // need to analyze why other (smaller) components are separated
    // => so we join vertices with original concept graph (original topicGraph with connected component graph)
    val topicComponentDF = topicGraph.vertices.innerJoin(
      connectedComponentGraph.vertices) {
      // inner join requires that we provide a function on vertexID
      // +++ and data contained inside of each of the two vertexRDD
      (topicId, name, componentId) => (name, componentId.toLong)
    }.values.toDF("topic", "cid")
    // look component which was not connected
    // TODO: insert our component id from df
    topicComponentDF.where("cid = -6468702387578666337").show()
    // we see the distribution of HIV topics
    val hiv = spark.sql("SELECT * FROM topic_dist WHERE topic LIKE '%hiv%'")
    hiv.show()
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

  /**
    * Use Google Guava library to get hashes for each vertex
    * (RDD[Long, String])
    * @param str
    * @return
    */
  def hashId(str: String) = {
    Hashing.md5().hashString(str).asLong()
  }
}

class GraphRun(private val spark: SparkSession) {

}
