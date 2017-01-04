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

    // to get inside how graph is structured
    // => better find degree of each vertex (number of edges that a particular vertex belongs to)
    val degrees: VertexRDD[Int] = topicGraph.degrees.cache()
    // calculate stats (mean, std, count, min-max) to see distribution of degrees
    // NOTE: degrees have smaller number of entries
    // WHY? some vertices have no edges
    // ---
    // ANALYSIS:
    // mean (49) - average vertex connected only to small fraction
    // max (3000) - there are some highly connected nodes
    degrees.map(_._2).stats()

    // look at the concepts
    degrees.innerJoin(topicGraph.vertices) {
      (topicId, degree, name) => (name, degree.toInt)
    }.values.toDF("topic", "degree").
      // order by degree to see the top connected vertices
      orderBy(desc("degree")).show()

    // Calculation of chi-squared test
    // T - total number of documents
    val T = medline.count()
    // create rdd from topic counts
    val topicDistRdd = topicDist.map { case Row(topic: String, cnt: Long) => (hashId(topic), cnt) }.rdd
    // create new graph from counts along with existing graph
    val topicDistGraph = Graph(topicDistRdd, topicGraph.edges)
    // need to combine data that is storead at both vertices
    // USE data structure called EdgeTriplet[VD, ED]
    // NOTE: returns new graph - edge attributes - are chi-sq test values
    val chiSquaredGraph = topicDistGraph.mapTriplets(triplet =>
      chiSq(triplet.attr, triplet.srcAttr, triplet.dstAttr, T)
    )
    // find stats (distribution) of new graph
    chiSquaredGraph.edges.map(x => x.attr).stats()
    // EXPLANATION why 19.5
    // 99.99 percentile of chi-sq distribution === 19.5 (x=19.5), where F_1(x) = 99.99
    // with one degree of freedom (k == 1)
    // FILTERING: use boolean to filter out
    val interesting = chiSquaredGraph.subgraph(triplet => triplet.attr > 19.5)
    val interestingComponentGraph = interesting.connectedComponents()
    val icDF = interestingComponentGraph.vertices.toDF("vid", "cid")
    val icCountDF = icDF.groupBy("cid").count()
    icCountDF.count()
    // overall: removed one third of all edges
    icCountDF.orderBy(desc("count")).show()

    // find degrees of filtered dataset
    val interestingDegrees = interesting.degrees.cache()
    // find their stats
    // CONCLUSION: can see that biggest component was not divided into smaller components
    // => CONNECTED GRAPH IS ROBUST TO NOISE!!
    interestingDegrees.map(_._2).stats()
    interestingDegrees.innerJoin(topicGraph.vertices) {
      (topicId, degree, name) => (name, degree)
    }.toDF("topic", "degree").orderBy(desc("degree")).show()

    // compute density graph metric - average clustering coefficent
    val avgCC = avgClusteringCoef(interesting)

    // compute average shortest path length using Pregel
    val paths = samplePathLengths(interesting)
    paths.map(_._3).filter(_ > 0).stats()

    // compute final stats in histogram
    val hist = paths.map(_._3).countByValue()
    hist.toSeq.sorted.foreach(println)
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

  /**
    * Calculation of chi-square test
    * @param YY
    * @param YB
    * @param YA
    * @param T
    * @return
    */
  def chiSq(YY: Long, YB: Long, YA: Long, T: Long): Double = {
    // create 2x2 table for any pair of concepts A and B
    // ELEMENTS: YY, YN, NY, NN - represent raw counts of presence/absence of concepts A and B
    // YA, NA - row sums for concept A
    // YB, NB - row sums for concept B
    // T - total number of documents
    val NB = T - YB
    val NA = T - YA
    val YN = YA - YY
    val NY = YB - YY
    val NN = T - NY - YN - YY
    // compute chi-square statistic from these values
    val inner = math.abs(YY * NN - YN * NY) - T / 2.0
    T * math.pow(inner, 2) / (YA * NA * YB * NB)
  }

  /**
    * Method to calculate average clustering coefficient metric
    * To show how dense is the graph
    * @param graph
    * @return
    */
  def avgClusteringCoef(graph: Graph[_, _]): Double = {
    // returns a graph whose VertexRDD contains number of triangles for each vertex
    val triCountGraph = graph.triangleCount()
    // need to normalize these triangle counts by total number of possible triangles at each vertex
    val maxTrisGraph = graph.degrees.mapValues(d => d * (d - 1) / 2.0)
    // join vertexRDD of triangle counts rdd to vertexRdd of normalization terms
    // to normalization term vertices rdd
    val clusterCoefGraph = triCountGraph.vertices.innerJoin(maxTrisGraph) {
      (vertexId, triCount, maxTris) => if (maxTris == 0) 0 else triCount / maxTris
    }
    // compute average value of clustering coefficient
    clusterCoefGraph.map(_._2).sum() / graph.vertices.count()
  }

  def samplePathLengths[V, E](graph: Graph[V, E], fraction: Double = 0.02)
  : RDD[(VertexId, VertexId, Int)] = {
    val replacement = false
    // select 2% (fraction = 0.02) of VertexID values for our sample without replacement
    // 1729L - is a seed for random num generator
    val sample = graph.vertices.map(v => v._1).sample(
      replacement, fraction, 1729L)
    val ids = sample.collect().toSet
    // need to create new Graph object whose vertex Map[VertexId, Int] values:
    // are only nonempty if vertex is a member of sampled IDs
    val mapGraph = graph.mapVertices((id, v) => {
      if (ids.contains(id)) {
        Map(id -> 0)
      } else {
        Map[VertexId, Int]()
      }
    })

    // initial message is an empty map
    val start = Map[VertexId, Int]()
    // can call pregel method followed by update, iterate and mergeMaps
    // to execute during each iteration
    val res = mapGraph.ops.pregel(start)(update, iterate, mergeMaps)
    // once it completes we can flatMap the vertices to extract tuples (VId, VId, Int)
    // - values that represent the unique path lengths that were computed
    res.vertices.flatMap { case (id, m) =>
      m.map { case (k, v) =>
        if (id < k) {
          (id, k, v)
        } else {
          (k, id, v)
        }
      }
    }.distinct().cache()
  }

  /**
    * Used to merge the information from the new messages
    * into the state of the vertex
    * NOTE: both state and message are of the same type
    * @param m1 state of vertex
    * @param m2 message
    * @return
    */
  def mergeMaps(m1: Map[VertexId, Int], m2: Map[VertexId, Int]): Map[VertexId, Int] = {
    def minThatExists(k: VertexId): Int = {
      math.min(
        m1.getOrElse(k, Int.MaxValue),
        m2.getOrElse(k, Int.MaxValue))
    }
    // merge state and message into new state
    // and retain smallest value associated with any vertexId entries that occur in both maps
    (m1.keySet ++ m2.keySet).map(k => (k, minThatExists(k))).toMap
  }

  /**
    * Wrapper arount mergeMaps function
    * @param id
    * @param state
    * @param msg
    * @return
    */
  def update(id: VertexId, state: Map[VertexId, Int], msg: Map[VertexId, Int])
  : Map[VertexId, Int] = {
    mergeMaps(state, msg)
  }


  def checkIncrement(a: Map[VertexId, Int], b: Map[VertexId, Int], bid: VertexId)
  : Iterator[(VertexId, Map[VertexId, Int])] = {
    // each vertex should increment the value of each key in its current Map{vertexId, int] by one
    val aplus = a.map { case (v, d) => v -> (d + 1) }
    // combine the incremented map values with the values from its neighbor using mergeMaps
    if (b != mergeMaps(aplus, b)) {
      // send the result of mergemaps to neighboring vertex
      // if it is different from neighbors internal map
      //
      Iterator((bid, aplus))
    } else {
      Iterator.empty
    }
  }

  /**
    * Used for performing message updates at each pregel iteration
    * @param e
    * @return
    */
  def iterate(e: EdgeTriplet[Map[VertexId, Int], _]): Iterator[(VertexId, Map[VertexId, Int])] = {
    checkIncrement(e.srcAttr, e.dstAttr, e.dstId) ++
      checkIncrement(e.dstAttr, e.srcAttr, e.srcId)
  }
}

class GraphRun(private val spark: SparkSession) {

}
