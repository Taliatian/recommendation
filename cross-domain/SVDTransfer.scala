package main.scala

import breeze.numerics.abs
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.{SparkConf, SparkContext}
import test.scala.LFMtest.{LearningLFM, computeRmse}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by tian on 2018/10/15.
  */

//思路：Rs = U*Ss*Vs
//     Rt = U*St*Vt
//根据用户重叠可以迁移用户特征矩阵，中间矩阵保留各自领域的特有特征

object SVDTransfer {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("SVDT")
    val sc = new SparkContext(conf)
    val sTrainData = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.base")
    val sTrainRatings = sTrainData.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))

    //构建评分矩阵
    var sRatingEntry = sTrainRatings.map(each => MatrixEntry(each._1,each._2,each._3))

    //计算用户和项目的个数
    val userCount = sTrainRatings.map(_._1).max + 1
    val itemCount = sTrainRatings.map(_._2).max + 1
    //print("user" + userCount + "\n" + "item" + itemCount) 943,1650

    //SVD分解,使用坐标矩阵是因为矩阵稀疏
    val Fs = 30 //F是特征数
    var sRatingMatrix = (new CoordinateMatrix(sRatingEntry,userCount,itemCount)).toRowMatrix()
    val sSVD = sRatingMatrix.computeSVD(Fs,computeU = true)
    val U = sSVD.U.rows
    val Vs = sSVD.V.transpose.toArray
    val Ss = sSVD.s.toArray
//    val Vt = Vs  //此处为了测试将源领域的V也迁移过去
    println("Vs的长度为 " + Vs.length)

    //此处是为了将u的类型转化成可以相乘的类型
    val userS = sTrainData.map(line => line.split("\t")).map(line => line(0).toInt).distinct().collect()
//    val itemRDD = sTrainData.map(line => line(1)).distinct().persist()
    val U1 = U.map(x => x.toArray).collect()
    //U2(user,userfeatures)即之后需使用的
    var U2 = Map[Int,Array[Double]]()
//    println(user.length + " " + U1.length)
    var i = 0
    for (user <- userS){
      U2 += (user -> U1(i))  //此处要不要转化成buffer
      i = i + 1
    }
//    U2.values.foreach(println)
//    U2.foreach(println)


    //需要搞清楚三个矩阵的类型
//    - U is a RowMatrix of size m x k that satisfies U' * U = eye(k),
//    - s is a Vector of size k,  the singular values in descending order,
//    - V is a Matrix of size n x k that satisfies V' * V = eye(k)
//    U.foreach(println)
//    println(Vs)
//    print(Ss)


    //以下是svd分解
    //目标领域进行SVD分解
    val tTrainData = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u2.base")
    val tTrainRatings = tTrainData.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    var tRatingEntry = tTrainRatings.map(each => MatrixEntry(each._1,each._2,each._3))
    val userCount1 = tTrainRatings.map(_._1).max + 1
    val itemCount1 = tTrainRatings.map(_._2).max + 1
    val Ft = 30
    var tRatingMatrix = (new CoordinateMatrix(tRatingEntry,userCount1,itemCount1)).toRowMatrix()
    val tSVD = sRatingMatrix.computeSVD(Ft,computeU = true)
    val Vt = tSVD.V.transpose.toArray  //Vt是项目特征矩阵的数组化
    val St = tSVD.s.toArray
//    如何将一维数组变成二维数组
//    var Vt1 = Array[Array[Double]]() 1683是项目的个数
    var Vt1 = Array.ofDim[Double](1683,30)

    var j = 0
    for(i <- Range(0,Vt.length-1,Ft)) {
      var tmp = new Array[Double](30)
       System.arraycopy(Vt, i, Vt1(j), 0, 30)
//       println(Vt1(j))
       j = j + 1
    }
      //    Vt.foreach(println)
    println("Vt的长度为 " + Vt.length)  //16830
    val itemT = tTrainData.map(line => line.split("\t")).map(line => line(1).toInt).distinct().collect()
    //将项目矩阵转化成（item，features）键值对
    var Vt2 = Map[Int, Array[Double]]()
    var ii = 0
    for (item <- itemT) {
        Vt2 += (item -> Vt1(ii) )
        ii = ii + 1
      }

//    val ITERATIONS = 100  //迭代次数
//    val LAMBDA = 0.01  //正则参数
//    val UVt = LearningLFM(tTrainRatings,Ft,ITERATIONS,LAMBDA)
//    val Vt3 = UVt(1)

      //整理St的结构为矩阵
    var StP = Array.ofDim[Double](30,30)
      for(i <- 0 to 29){
        for(j <- 0 to 29){
          if(j == i)
            StP(i)(j) = St(j)
          else
            StP(i)(j) = 0
        }
      }
      //在目标领域进行迁移U
    val predictRating = for ((k1, v1) <- U2; (k2, v2) <- Vt2) yield {
      var sum = 0.0
      for(k <- 0 until Ft){
        sum += v1(k) * StP(k)(k) * v2(k)
//        println(sum)
        sum = abs(sum)

//        sum = sum + 1  //有待商榷
//        println(sum)
        if(sum < 1) sum = 1
        if(sum > 5) sum = 5
      }
      (k1,k2,sum)
    }

//    val predictRating = for ((k1, v1) <- U2; (k2, v2) <- Vt3) yield {
//      var sum = 0.0
//      for(k <- 0 until Ft){
//        sum += v1(k) * StP(k)(k) * v2(k)
//        //        println(sum)
////        if(sum < 0) sum = 0
//      }
//      (k1,k2,sum)
//    }
    val predictRatingFinal = predictRating.toList
    val predictRatingRDD = sc.parallelize(predictRatingFinal)
    //提取测试集
    val testT = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u2.test")
    val numTest = testT.count()
    val testTRating = testT.map(_.split("\t")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val Rmse = computeRmse(predictRatingRDD,testTRating,numTest)
    println("计算出的RMSE值为： " + Rmse)
  }
}

