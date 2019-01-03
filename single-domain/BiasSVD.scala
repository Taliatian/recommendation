package main.scala

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
//import test.scala.LFMtest.computeRmse
import  main.scala.Evaluation._

import scala.collection.mutable
import scala.collection.mutable.Buffer
import scala.collection.mutable.Map

/**
  * Created by tian on 2018/11/21.
  */
object BiasSVD {

  def initBiasSVD(dataRdd: RDD[(Int, Int, Double)], F: Int) = {
    val userRdd = dataRdd.map(each => each._1.toInt).distinct().persist()
    val itemRdd = dataRdd.map(each => each._2.toInt).distinct().persist()
//    println(userRdd.count())
    val pInit = userRdd.map(each => (each,(for(i <- Range(0,F)) yield (math.random / math. sqrt(F))))).collect()
    val qInit = itemRdd.map(each => (each,(for(i <- Range(0,F)) yield (math.random / math. sqrt(F))))).collect()
    //    pInit.foreach(println)
    val buInit = userRdd.map(each => (each,0.0)).collect()
    val biInit = itemRdd.map(each => (each,0.0)).collect()
    var P = Map[Int,Buffer[Double]]()
    var Q = Map[Int,Buffer[Double]]()
    for(x <- pInit){
      P = P + (x._1 -> x._2.toBuffer)
    }
    for(x <- qInit){
      Q = Q + (x._1 -> x._2.toBuffer)
    }
    val value = (P,Q,buInit,biInit)
    value
    //        val t = ArrayBuffer(for(i <- Range(0,F)) yield (math.random / math. sqrt(F)))  //测试
  }

  def predict(user: Int, item: Int, P1: Map[Int, mutable.Buffer[Double]], Q1: mutable.Map[Int, mutable.Buffer[Double]],bu2: mutable.Map[Int,Double], bi2: Map[Int,Double], mu: Double, F: Int): Double = {
//    var sum: Double = 0.0

     var ret = bu2(user) + bi2(item) + mu
//    println(ret)
    for(i <- Range(0,F)){
      ret = ret + P1(user)(i) * Q1(item)(i)
    }
    ret
  }

  def learningBiasSVD(dataRdd: RDD[(Int, Int, Double)], F: Int, alpha: Double, interation: Int, lambda: Double, mu: Double) = {
    val PQ = initBiasSVD(dataRdd,F)
    val P1 = PQ._1
    val Q1 = PQ._2
    val bu1 = PQ._3
    val bi1 = PQ._4
//    val bu2 = bu1.foreach(x => Map(x._1 -> x._2))
    var bu2 = Map[Int,Double]()
    for(x <- bu1){
      bu2 = bu2 + (x._1 -> x._2)
    }
    var bi2 = Map[Int,Double]()
    for(x <- bi1){
      bi2 = bi2 + (x._1 -> x._2)
    }

    var alpha1 = alpha
    for(step <- Range(0,interation)){
      for(x <- dataRdd.collect()){
        val pui = predict(x._1,x._2,P1,Q1,bu2,bi2,mu,F)
        //        println(pui)
        val eui = x._3 - pui
        bu2(x._1) = bu2(x._1) + alpha1 * (eui - lambda * bu2(x._1))
        bi2(x._2) = bi2(x._2) + alpha1 * (eui - lambda * bi2(x._2))
        for(f <- Range(0,F)){
          P1(x._1)(f) += alpha1 * (Q1(x._2)(f) * eui - lambda * P1(x._1)(f))
          Q1(x._2)(f) += alpha1 * (P1(x._1)(f) * eui - lambda * Q1(x._2)(f))
        }
      }
      alpha1 = alpha1 *  0.9
      println("循环第" + step + "次")
    }
    val value = (P1,Q1,bu2,bi2)
    value
  }

//  def computeRmse(predictRatings: RDD[(Int, Int, Double)], testRatings: RDD[(Int, Int, Double)], numTest: Long): Double = {
//    val predictTestRatings = predictRatings.map(x => ((x._1,x._2),x._3)).join(testRatings.map(x => ((x._1,x._2),x._3))).values
//    val rmse = math.sqrt(predictTestRatings.map(x => (x._2 - x._1) * (x._2 - x._1)).sum() / numTest)
//    rmse
//  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BiasSVD").setMaster("local")
    val sc = new SparkContext(conf)
//    val data = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.base")
    val data = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.base")
    val dataRdd = data.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val dataRating = data.map(_.split("\t")).map(line => line(2).toDouble)
    val mu = dataRating.sum() / dataRating.count()
    println("平均值" + mu)
    val F = 50
    val interation = 50
    val lambda = 0.01
    val alpha = 0.01
    val UV = learningBiasSVD(dataRdd,F,alpha,interation,lambda,mu)
    val U = UV._1
    val V = UV._2
    val bu = UV._3
    val bi = UV._4
    val predictrating = for ((k1,v1) <- U; (k2,v2) <- V) yield {
      var ret = bu(k1) + bi(k2) + mu
      for (k <- 0 until F) {
        ret += v1(k) * v2(k)
      }
      (k1, k2, ret)
    }
    println("读入测试文件⋯⋯")
//    val test = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.test")
    val test = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.test")
    val numTest = test.count()
    println("测试集的长度： " + numTest)
    val testRatings = test.map(_.split("\t")).map(line => (line(0).toInt, line(1).toInt, line(2).toDouble))
//    testRatings.take(5).foreach(println)
    val predictratings = sc.parallelize(predictrating.toList)
    println("计算RMSE值")
    val Rmse = computeRmse(predictratings, testRatings)
    println("计算得出RMSE值为： " + Rmse)
    println("计算MAE值")
    val MAE = computeMAE(predictratings, testRatings)
    println("计算得出MAE值为： " + MAE)
//    val preRec = precisionRecall(predictratings,testRatings,10)
//    println("准确率： " + preRec._1 + " " + "召回率： " + preRec._2)
  }

}
