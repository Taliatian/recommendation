package test.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
//import test.scala.LFMtest.computeRmse
import main.scala.Evaluation._

import scala.collection.mutable
import scala.collection.mutable.Buffer

/**
  * Created by tian on 2018/11/15.
  */
object LFM {

  def initLFM(dataRdd: RDD[(Int, Int, Double)], F: Int): List[Map[Int,Buffer[Double]]] = {
    val userRdd = dataRdd.map(each => each._1.toInt)
    val itemRdd = dataRdd.map(each => each._2.toInt)
    val pInit = userRdd.map(each => (each,(for(i <- Range(0,F)) yield (math.random / math. sqrt(F))))).collect()
    val qInit = itemRdd.map(each => (each,(for(i <- Range(0,F)) yield (math.random / math. sqrt(F))))).collect()
//    pInit.foreach(println)
    var P = Map[Int,Buffer[Double]]()
    var Q = Map[Int,Buffer[Double]]()
    for(x <- pInit){
      P = P + (x._1 -> x._2.toBuffer)
    }
    for(x <- qInit){
      Q = Q + (x._1 -> x._2.toBuffer)
    }
    List(P,Q)
//        val t = ArrayBuffer(for(i <- Range(0,F)) yield (math.random / math. sqrt(F)))  //测试
  }

  def predict(user: Int, item: Int, P1: Map[Int, mutable.Buffer[Double]], Q1: Map[Int, mutable.Buffer[Double]], F: Int): Double = {
    var sum: Double = 0.0
    for(i <- Range(0,F)){
      sum = sum + P1(user)(i) * Q1(item)(i)
    }
    sum
  }

  def learningLFM(dataRdd: RDD[(Int, Int, Double)], F: Int, alpha: Double, interation: Int, lambda: Double): List[Map[Int,Buffer[Double]]] = {
    val PQ = initLFM(dataRdd,F)
    val P1 = PQ(0)
    val Q1 = PQ(1)
    var alpha1 = alpha
    for(step <- Range(0,interation)){
      for(x <- dataRdd.collect()){
        val pui = predict(x._1,x._2,P1,Q1,F)
//        println(pui)
        val eui = x._3 - pui
        for(f <- Range(0,F)){
          P1(x._1)(f) += alpha1 * (Q1(x._2)(f) * eui - lambda * P1(x._1)(f))
          Q1(x._2)(f) += alpha1 * (P1(x._1)(f) * eui - lambda * Q1(x._2)(f))
        }
      }
      alpha1 = alpha1 *  0.9
      println("循环第" + step + "次")
    }
    List(P1,Q1)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LMF").setMaster("local")
    val sc = new SparkContext(conf)
    val data = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/douban/movie整理train")
    val dataRdd = data.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val F = 30
    val interation = 20
    val lambda = 0.01
    val alpha = 0.01
    val UV = learningLFM(dataRdd,F,alpha,interation,lambda)
    val U = UV(0)
    val V = UV(1)
    val predictrating = for ((k1,v1) <- U; (k2,v2) <- V) yield {
      var sum = 0.0
      for (k <- 0 until F) {
        sum += v1(k) * v2(k)
      }
      (k1, k2, sum)
    }
    println("读入测试文件⋯⋯")
    val test = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/douban/movie整理test")
    val numTest = test.count()
    val testrating = test.map(_.split(",")).map(line => (line(0).toInt, line(1).toInt, line(2).toDouble))
    val predictratings = sc.parallelize(predictrating.toList)
//    predictratings.filter(_._1 == 1).sortBy(_._3,false).take(5).foreach(println)
//    testrating.filter(_._1 == 1).sortBy(_._3,false).take(100).foreach(println)
//    println(testrating.filter(_._1 == 1).count() + " " + predictratings.filter(_._1 == 1).count())

//    println("计算RMSE值")
    val Rmse = computeRmse(predictratings, testrating)
    println("计算得出RMSE值为： " + Rmse)
//    println("计算MAE值")
//    val MAE = computeMAE(predictratings,testrating)
//    println("计算得出MAE值为： " + MAE)
//    println("正在计算准确率和召回率---")
//    val preRec = PrecisionRecall(predictratings,testrating,10)
//    println("准确率： " + preRec._1 + " " + "召回率： " + preRec._2)
  }

}
