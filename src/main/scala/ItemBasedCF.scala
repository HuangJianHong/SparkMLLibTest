import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 建立物品的相似度，来进行推荐
  *
  * 需求： 为某一个用户推荐商品。基本的逻辑是：首先得到某个用户评价过（买过的商品），
  * 然后计算其他商品与改商品的相似度，并排序；从高到低，把不在用户评价过的商品
  * 里的其他商品推荐给用户
  *
  */
object ItemBasedCF {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //读入数据
    val conf = new SparkConf().setAppName("UserBaseModel").setMaster("local")
    val sc = new SparkContext(conf)
    val data = sc.textFile("D:\\temp\\data\\ratingdata.txt")

    /**
      * MatrixEntry代表一个分布式矩阵中的每一行(Entry)
      * 这里的每一项都是一个(i: Long, j: Long, value: Double) 指示行列值的元组tuple。
      * 其中i是行坐标，j是列坐标，value是值。
      **/
    val parseData: RDD[MatrixEntry] =
      data.map(_.split(",")
      match { case Array(user, item, rate) => MatrixEntry(user.toLong, item.toLong, rate.toDouble)
      })

    //CoordinateMatrix是 Spark MLLib中专门保存 user_item_rating这种数据样本
    val ratings = new CoordinateMatrix(parseData)

    /**
      * RowMatrix的方法columnSimilarities是计算，列与列的相似度;
      */
    val matrix: RowMatrix = ratings.toRowMatrix()

    /**
      * 需求： 为某一个用户推荐商品。基本的逻辑是：首先得到某个用户评价过（买过的商品），
      * 然后计算其他商品与改商品的相似度，并排序；从高到低，把不在用户评价过的商品
      * 里的其他商品推荐给用户
      *
      * 例如：为用户2推荐商品
      */

    /**
      * 得到用户2评价过（买过）的商品，take(5)表示取出所有5个用户 2：表示第二个用户
      * 解释：SparseVector：稀疏矩阵
      */
    val user2pred: linalg.Vector = matrix.rows.take(5)(2)
    val prefs: SparseVector = user2pred.asInstanceOf[SparseVector]
    val uitems: Array[Int] = prefs.indices //得到用户2评价过的（买过）的商品ID

    //得到了用户2评价过（买过）的商品的ID和评分，即（物品ID,评分）
    val ipi: Array[(Int, Double)] = uitems zip prefs.values
    for (s <- ipi) println(s)
    println("**********")

    //计算物品的相似度， 并输出
    val similarities: CoordinateMatrix = matrix.columnSimilarities()
    val indexdsimilar: RDD[(Int, linalg.Vector)] = similarities.toIndexedRowMatrix().rows.map {
      case IndexedRow(index, vector) => (index.toInt, vector)
    }
    indexdsimilar.foreach(println)
    println("*************")

    //ij表示:其他用户购买的商品与用户2购买的商品的相似性
    val ij: RDD[(Int, Double)] = sc.parallelize(ipi).join(indexdsimilar).flatMap {
      case (i, (pi, vector: SparseVector)) => (vector.indices zip vector.values)
    }


    //ij1表示：其他用户购买过，但不在用户2购买的商品的列表中的商品 和 评分
    val ij1: RDD[(Int, Double)] = ij.filter { case (item, pref) => !uitems.contains(item) }
    ij1.foreach(println)
    println("**********")

    //将这些商品 的评分求和，并降序排列，并推荐前两个物品
    val ij2: Array[(Int, Double)] = ij1.reduceByKey(_ + _).sortBy(_._2, false).take(2)
    println("*******推荐的结果是：******")
    ij2.foreach(println)
  }

}
