package main.scala

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix
import test.scala.LFMtest.{LearningLFM, computeRmse}

import scala.collection.mutable

/**
  * Created by tian on 2018/11/12.
  */

//思路：1.根据源领域评分分解UV
//     2.计算用户的相似度矩阵
//     3.相似度矩阵进行分解得到用户特征矩阵
//     4.进行迁移至目标领域


object USMFT {

//  def userCosineSimilarity(model: MatrixFactorizationModel) = {
//    val userVectorRdd = model.userFeatures.map{
//      case (userId,userFactor) => val userFactorVector = new DoubleMatrix(userFactor)
//        (userId,userFactorVector)
//    }
//    val userSimilarity = userVectorRdd.cartesian(userVectorRdd).filter{
//      case ((userId1,vector1),(userId2,vector2)) => userId1 != userId2
//    }
//    println(userSimilarity.count())
//  }


  def cosineSimilarity(vector1: DoubleMatrix, vector2: DoubleMatrix): Double = vector1.dot(vector2) / (vector1.norm2() * vector2.norm2())

  def userCosineSimilarity(userFactorRdd: RDD[(Int, Array[Double])]) = {
    val userVectorRdd = userFactorRdd.map{
      case (userId,factor) => val userFactorVector = new DoubleMatrix(factor)
        (userId,userFactorVector)
    }
    val userSimilarity1 = userVectorRdd.cartesian(userVectorRdd).filter{
      case ((user1,vector1),(user2,vector2)) => user1 != user2
    }.map{
      case ((user1,vector1),(user2,vector2)) => val sim1 = cosineSimilarity(vector1,vector2)
        (user1,user2,sim1)
    }
    val userSimMin = userSimilarity1.map(_._3).min()
    val userSimMax = userSimilarity1.map(_._3).max()
    println("相似度最大值" + userSimMax)
    println("相似度最小值" + userSimMin)
//    此处为了使相似度矩阵的值与目标领域评分矩阵值一致
    val interval = (userSimilarity1.map(_._3).max() + 1 - userSimilarity1.map(_._3).min()) / 5
    val userSimilarity = userSimilarity1.map{
      case (user1,user2,sim) =>
        val sim2 = {
          if((sim >= userSimMin) && (sim < (userSimMin + interval))) 1
          else if((sim >= (userSimMin + interval)) && (sim < userSimMin + 2 * interval)) 2
          else if((sim >= (userSimMin + 2 * interval)) && (sim < userSimMin + 3 * interval)) 3
          else if((sim >= (userSimMin + 3 * interval)) && (sim < userSimMin + 4 * interval)) 4
          else 5
        }
        (user1,user2,sim2.toDouble)
    }
//    userSimilarity.take(100).foreach(println)

    //    println("笛卡尔积" + userSimilarity.count())
//    println(userVectorRdd.cartesian(userVectorRdd).count())
    userSimilarity
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("USMFT")
    val sc = new SparkContext(conf)

//    读数据集
    val sTrainData = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.base")
    val tTrainData = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u2.base")
    val tTestData = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u2.test")
    val sTrain = sTrainData.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble)).cache()

//    使用als出现栈溢出 什么问题？稍后解决

////    将源域进行ALS矩阵分解
//    val rank = 20
//    val iterations = 30
//    val lambda = 0.01
//    val sourceModel = ALS.train(sTrain, rank, iterations, lambda)
//
////    计算用户相似度矩阵
////    val sourceUserSim = userCosineSimilarity(sourceModel)
//    println("ok")

    val users = sTrain.map(_._1).distinct()
    println("用户个数" + users.count())
//    使用LFM
    val F = 100  //LFM的特征数
    val ITERATIONS = 70  //迭代次数
    val LAMBDA = 0.0001  //正则参数
    val UVs = LearningLFM(sTrain,F,ITERATIONS,LAMBDA)
    val userFactor = UVs(0).map(x => (x._1,x._2.toArray)).toList
    val userFactorRdd = sc.parallelize(userFactor)
    val sourceUserSim = userCosineSimilarity(userFactorRdd)

//    分解相似度矩阵
    val SourceSimMatrixFac = LearningLFM(sourceUserSim,F,5,LAMBDA)
    val SourceSimU = SourceSimMatrixFac(0)

//    处理目标领域
    val tRatings = tTrainData.map(line => line.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble)).coalesce(1)
    val UVt = LearningLFM(tRatings,F,ITERATIONS,LAMBDA)
    val Vt = UVt(1)
    val predictRating = for((k1,v1) <- SourceSimU; (k2,v2) <- Vt) yield {
      var sum = 0.0
      for(k <- 0 until F) {
        sum = sum + v1(k) * v2(k)
        //println(sum)
      }
      (k1,k2,sum)
    }
    val predictRatingFinal = predictRating.toList
    //    predictRatingFinal.filter(_._1 == 1).take(100).foreach(println)
    val predictRatings = sc.parallelize(predictRatingFinal)
    val numTest = tTestData.count()
    val testRating = tTestData.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val Rmse = computeRmse(predictRatings,testRating,numTest)
    println("计算得出RMSE值为： " + Rmse)
  }
}
