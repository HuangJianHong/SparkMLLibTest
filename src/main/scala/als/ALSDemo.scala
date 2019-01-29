package als

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object ALSDemo {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //读入数据，并转化为RDD[Rating]，得到评分数据
    val conf: SparkConf = new SparkConf().setAppName("ALSDemo").setMaster("local")
    val sc = new SparkContext(conf)


    val productRatings: Seq[Rating] = loadRatingData("D:\\temp\\data\\ratingdata.txt")
    val prodcutRatingsRDD: RDD[Rating] = sc.parallelize(productRatings)
    //    val value: RDD[String] = sc.textFile("D:\\temp\\data\\ratingdata.txt")


    //输出测试信息
    val numRatings: Long = prodcutRatingsRDD.count()
    val numUsers: Long = prodcutRatingsRDD.map(x => x.user).distinct().count()
    val numProducts: Long = prodcutRatingsRDD.map(x => x.product).distinct().count()
    println("评分数：" + numRatings + "\t 用户总数：" + numUsers + "\t 物品总数：" + numProducts)


    /** 查看ALS训练模型的API
      *ALS.train(ratings, rank, iterations, lambda)
      * 参数说明：ratings：评分矩阵
      * rank：小矩阵中，特征向量的个数。推荐的经验值：建议： 10~200之间
      * rank越大，表示：拆分越准确
      * rank越小，表示：速度越快
      * *
      * iterations:运行时的迭代（循环）次数，经验值：10左右
      * lambda：控制拟合的正则化过程，值越大，表示正则化过程越厉害；如果这个值越小，越准确 ，使用0.01
      */
    val model: MatrixFactorizationModel = ALS.train(prodcutRatingsRDD, 50, 10, 0.01)
    val rmse: Double = computeRMSE(model, prodcutRatingsRDD, numRatings)
    println("误差：" + rmse)

    //使用该模型，来进行推荐
    //需求：给用户1推荐2个商品
    val recomm: Array[Rating] = model.recommendProducts(1, 2)
    recomm.foreach(r => {
      println("用户：" + r.user.toString() + "\t 物品：" + r.product.toString() + "\t 评分:" + r.rating.toString())
    })


    sc.stop()
  }

  /**
    * 加载评分数据
    *
    * @param path 评分数据本地路径
    * @return
    */
  def loadRatingData(path: String): Seq[Rating] = {
    val lines: Iterator[String] = Source.fromFile(path).getLines()

    //过滤掉评分数据是0的数据
    val ratings: Iterator[Rating] = lines.map(line => {
      val fields: Array[String] = line.split(",")
      //返回Rating的对象：用户ID、物品ID、评分数据
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }).filter(x => x.rating > 0.0)

    //转化成Seq[Rating]
    if (ratings.isEmpty) {
      sys.error("Error....")
    } else {
      //返回 Seq[Rating]
      ratings.toSeq
    }
  }


  /**
    * 计算RMSE ： 均方根误差；
    * 判断model是否是 最优推荐模型
    */
  def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    //真实值：
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))

    //数据：真实值((userID,itemID),rating) join 观测值 ((userID,itemID),rating)
    /*
     * select 真实值的评分,观测值的评分
     * from 真实值,观测值
     * where 真实值.(userID,itemID) = 观测值.(userID,itemID)
     */
    val predictionsAndRating = predictions.map {
      x => ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values

    //    println("******************************")
    //    predictionsAndRating.foreach(println)
    //    println("******************************")
    math.sqrt(predictionsAndRating.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

}
