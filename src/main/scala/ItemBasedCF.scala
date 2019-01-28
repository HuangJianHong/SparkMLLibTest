import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 建立物品的相似度，来进行推荐
  */
object ItemBasedCF {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //读入数据
    val conf = new SparkConf().setAppName("UserBaseModel").setMaster("local")
    val sc = new SparkContext(conf)
    val data = sc.textFile("D:\\download\\data\\ratingdata.txt")

    /**
      * MatrixEntry代表一个分布式矩阵中的每一行(Entry)
      * 这里的每一项都是一个(i: Long, j: Long, value: Double) 指示行列值的元组tuple。
      * 其中i是行坐标，j是列坐标，value是值。
      **/
    val parseData: RDD[MatrixEntry] =
      data.map(_.split(",")
      match { case Array(user, item, rate) => MatrixEntry(user.toLong, item.toLong, rate.toDouble)
      })




  }

}
