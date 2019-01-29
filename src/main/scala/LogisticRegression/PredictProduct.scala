package LogisticRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 逻辑回归： 二分类，预测用户行为 买或者不买
  *
  *
  *
  * 1、如何代表人工标注的数据？
  * case class LabeledPoint(label: Double, features: Vector)
  * 参数：label: 人工标注的数据
  * features: 特征向量
  *
  * 2、如何进行逻辑回归？
  * LogisticRegressionWithLBFGS: 支持多分类，可以更快的聚合（速度快、效果好）
  * LogisticRegressionWithSGD  : 梯度下降法，只支持二分类。在Spark 2.1后，基本被废弃了
  */

object PredictProduct {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf: SparkConf = new SparkConf().setMaster("local").setAppName("PredictProduct")
    val sc = new SparkContext(conf)

    val data: RDD[String] = sc.textFile("D:\\temp\\data\\sales.data")
    //读入数据 ----> 封装LabeledPoint
    val parseData: RDD[LabeledPoint] = data.filter(_.length > 0).map(line => {
      //分词操作：按照@分词
      val parts: Array[String] = line.split("@")
      //生成一个LabeledPoint(label: Double, features: Vector)
      //             人工标注的数据         该行的特征向量
      LabeledPoint(parts(0).trim.toDouble, Vectors.dense(parts(1).split(",").map(_.trim.toDouble)))
    })

    //执行逻辑回归:   2 表示二分类，
    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(parseData)

    /**
      * 生产上，将模型保存到HDFS，并不需要经常变换
      * HDP的端口 8020；
      * 需要设置，环境变量，否则权限不过，运行有问题
      */
//    System.setProperty("HADOOP_USER_NAME", "hdfs")
//    model.save(sc, "hdfs://192.168.18.21:8020/PredictMode")
//
//    模型已经存的话， 加载改模型
//    model: LogisticRegressionModel = LogisticRegressionModel.load(sc, "hdfs://192.168.18.21:8020/PredictMode")


    //使用该模型进行预测用户
    /* 用户的特征数据：
     * 1,23,175,6  ------>  买还是不买？1.0
     * 0,22,160,5  ------>  买还是不买？0.0
     */
    //    val target: linalg.Vector = Vectors.dense(1, 23, 175, 6)
    val target: linalg.Vector = Vectors.dense(0, 22, 160, 5)

    //执行预测
    val result: Double = model.predict(target)
    println("用户 买还是不买？" + result)


    sc.stop();
  }

}
