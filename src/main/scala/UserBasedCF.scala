import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object UserBasedCF {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //创建一个SparkContext
    val conf = new SparkConf().setAppName("BlackUserList").setMaster("local")
    val sc = new SparkContext(conf)

    //读入数据
    val data = sc.textFile("D:\\temp\\data\\ratingdata.txt")

    System.out.println("============")
    sc.stop();
  }
}
