package main.scala

import breeze.numerics.abs
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
  * Created by tian on 2018/12/5.
  */
object Evaluation {
  def computeRmse(prediction:RDD[(Int, Int, Double)],test:RDD[(Int, Int, Double)]): Double = {
    val predictionAndRating = prediction.map{ line => ((line._1, line._2), line._3)}
      .join(test.map(line => ((line._1, line._2), line._3))).values
//    predictionAndRating.foreach(println)
    val num = predictionAndRating.count()
    val num2 = test.count()
    println("训练集和测试集共有长度： " + num + "测试集长度： " + num2)
    math.sqrt(predictionAndRating.map( x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / num)
  }

  def computeMAE(prediction:RDD[(Int,Int,Double)], test:RDD[(Int,Int,Double)]): Double = {
    val predictionAndRating = prediction.map{line => ((line._1,line._2),line._3)}
      .join(test.map(line => ((line._1,line._2),line._3))).values
//    predictionAndRating.foreach(println)
    val num = predictionAndRating.count()
    val num2 = test.count()
    println("训练集和测试集共有长度： " + num)
    val MAE = predictionAndRating.map(x => abs(x._1 - x._2)).reduce(_ + _) / num2
    MAE
  }

  def recommendation(rating:RDD[(Int,Int,Double)], user: Int, n:Int) = {
    val items = rating.filter(_._1 == user).sortBy(_._3,false).take(n).map(x => x._2)
    val userItems = (user,items)
    userItems
  }

  def itemsSet(user: Int, rating: RDD[(Int, Int, Double)]) = {
    val items = rating.filter(_._1 == user).map(x => x._2)
    val userItems = (user,items)
    userItems
  }

  def precisionRecall(prediction:RDD[(Int,Int,Double)], test:RDD[(Int,Int,Double)], N: Int) ={
    val preUsers = prediction.map(x => x._1).distinct().collect()
    val testUsers = test.map(x => x._1).distinct().collect()
    var hit = 0.0
    var nRecall = 0.0
    var nPrecision = 0.0
//    val N = 10
    for(x <- testUsers){
      if(preUsers.contains(x)){
//        println(x)
//        函数返回值应该是（一个用户，项目集合）
        val testUserItems = recommendation(test,x,N)
        val predictionUserItems = recommendation(prediction,x,N)
        val preTestItemIntersect = (x,predictionUserItems._2 intersect testUserItems._2)
        hit += preTestItemIntersect._2.length
        val itemsRecallDeno = itemsSet(x,test)
        val itemsRecall = itemsRecallDeno._2.count()
        println("用户" + " " + x)
        println("训练集和测试集共有的item长度" + hit)
        nRecall += itemsRecall
        println("recall分母： " + itemsRecall)
        nPrecision += N
        println("precision分母： " + N)
      }
    }
    println("hit:" + " " + hit)
    println("nRecall:" + " " + nRecall)
    println("nPrecision:" + " " + nPrecision)
    val Recall = hit / nRecall
    val Precision = hit / nPrecision
    (Precision,Recall)
  }


//  测试准确率和召回率
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Test")
    val sc = new SparkContext(conf)
    val data1 = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.base")
    val data2 = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.test")
    val rdd1 = data1.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val rdd2 = data2.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val value = precisionRecall(rdd1,rdd2,5)
    print(value._1 + " " + value._2)
  }


}
