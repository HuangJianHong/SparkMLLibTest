import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


/**
  * 1、建立用户的相似度矩阵: 注意是用户和用户之间的
  * 2、计算一些数据: 用户相似度，平均评分，对所有物品的评分
  */
object UserBasedCF {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //创建一个SparkContext
    val conf = new SparkConf().setAppName("BlackUserList").setMaster("local")
    val sc = new SparkContext(conf)

    //读入数据
    val data = sc.textFile("D:\\temp\\data\\ratingdata.txt")

    //MatrixEntry代表：矩阵中的一行
    //使用模式匹配:
    /**
      * MatrixEntry代表一个分布式矩阵中的每一行（Entry）
      * 这里的每一项 都是一个（i:Long， j:Long, value:Double） 指示行列值的元组tuple
      * 其中 i是行坐标，j是列坐标，value是值
      * 注意：坐标是从 0 开始的
      */
    val parseData: RDD[MatrixEntry] =
      data.map(_.split(",")
      match { case Array(user, item, rate) => MatrixEntry(user.toLong, item.toLong, rate.toDouble)
      })

    //构造评分矩阵
    /**
      * 由于CoordianteMatrix没有columnSimilarities方法，所以我们需要将其转换成RowMatrix矩阵，
      * 调用他的columnSimilarities计算其相似性
      *
      * RowMatrix的方法columnSimilarities是计算，列与列的相识度，现在是user_item_rating ,
      * 需要转置（transpose）成item_user_rating,这样才是用户的相似度
      */
    val ratings = new CoordinateMatrix(parseData)

    val matrix: RowMatrix = ratings.transpose().toRowMatrix()
    //计算用户的相似度
    val similarities: CoordinateMatrix = matrix.columnSimilarities()
    println("用户之间的相似度矩阵")
    similarities.entries.collect().map(x => {
      println(x.i + "->" + x.j + "->" + x.value)
    })

    //得到用户1: 用户id=1 对所有物品的评分
    val ratingOfUser1: Array[Double] = ratings.entries.filter(_.i == 1).map(x => {
      (x.j, x.value)
    }).sortBy(_._1).collect().map(_._2).toList.toArray
    println("得到用户1对每种物品的评分")
    for (s <- ratingOfUser1) println(s)

    //用户1 对所有物品的评价评分
    val avgRatingOfUser1: Double = ratingOfUser1.sum / ratingOfUser1.size
    println("用户1对所有物品的平均评分：" + avgRatingOfUser1)

    //其他用户对物品1的评分, drop(1)表示除去用户1的评分
    val otherRatingsToItem1: Array[Double] = matrix.rows.collect()(0).toArray.drop(1)
    println("其他用户对物品1的评分")
    for (s <- otherRatingsToItem1) println(s)


    //得到用户1相对于其他用户的相似性（即：权重），降序排列，value越大，表示相似度越高
    val weights: Array[Double] = similarities.entries.filter(_.i == 1).sortBy(_.value, false).map(_.value).collect()
    println("用户1相对于其他用户的相似性")
    for (s <- weights) println(s)


    sc.stop();
  }
}
