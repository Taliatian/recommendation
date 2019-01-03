package test.scala
import breeze.linalg.*
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
 import LFMtest.{LearningLFM, computeRmse}
import Array._
import scala.collection.mutable.{Buffer, Map}
/**
  * Created by tian on 2018/9/7.
  */
object CBT {

  def predict(n: Int, m: Int, U: Array[Array[Double]], V: DenseMatrix, B: Array[Double], f: Int):
  Double = {
    var sum: Double = 0
    for (k <- Range(0,f)){
      sum += U(n)(k) * B(k) * V.apply(m,k)
    }
    sum
  }

  def LearningCBT(n: Int, m: Int, rating: RDD[(Int, Int, Double)], ratings: RDD[MatrixEntry], U: Array[Array[Double]], V: DenseMatrix, f: Int, iter: Int, lambda: Double, B: Array[Double]):
  Array[Double] = {
    var CBT = B
    //println(B.length)
    var alpha = 0.001
    for (step <- 0 until iter) {
      //foreach用户遍历集合 map用于一个集合到另一个集合的映射
      rating.foreach { case (user, item, rating) => {
        val pui = predict(n,m,U,V,CBT,f)
        val eui = rating - pui
        for(k <- 0 until f){
          CBT(k) += alpha * ( CBT(k) * eui - lambda * CBT(k))
        }

      }
      }
      alpha = alpha *0.93
    }
    CBT
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("CBT")
    val sc = new SparkContext(conf)
    val trainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens100k用户重叠/sD/sD")
//    val testData = sc.textFile("/Users/tian/tjy/dataset/ml-100k/u1.test")
    val train1 = trainData.map(line => line.split(",")).map(line => (line(0).toInt, line(1).toInt, line(2).toDouble))
    //val train1 = trainData.map(line => line.split("\t")).map(line => (line(0).toInt, line(1).toInt, line(2).toDouble)).coalesce(1)

    val ratings = trainData.map(_.split(",")).map(line => (line(0).toInt, line(1).toInt, line(2).toDouble))
    val ratingEntry = ratings.map(each => MatrixEntry(each._1, each._2, each._3))
    val userCount = ratings.map(_._1).max + 1
//    val userCount = ratings.map(_._1).distinct().count() + 1
//    val itemCount = ratings.map(_._2).distinct().count() + 1

  //  print(userCount1,userCount2) //944 943
    val itemCount = ratings.map(_._2).max + 1
    //print(userCount,itemCount)
    val USERVECTOR = 10
    val ITEMVECTOR = 15
    //用户对项目的评分矩阵
    var ratingMatrix = (new CoordinateMatrix(ratingEntry, userCount, itemCount)).toRowMatrix()
    // println(ratingMatrix.rows.take(1).mkString(","))
    //第二个参数 是否计算矩阵U
    val SVD = ratingMatrix.computeSVD(10, computeU = true)
//    print(SVD.s) //输出对角线元素
    val U1 = SVD.U.rows //初始化U
    val V1 = SVD.V
    //print(U1)
    //print(V1)
    val B = SVD.s.toArray  //10*10
    //println(B.length) //10
    val arr = V1.toArray
    //println(arr.length)  //16830
   // println(V1.toArray.length) //16830
    //println(V1.transpose.toArray.length)
    //val arr2 = V1.toArray
    //print(arr2(1))
    //为什么步长是10,该段代码实现什么？
//    print(V1.transpose.numRows) //10
    for (i <- Range(0, arr.length, 10)) {
      var tmp = new Array[Double](10)
      val subArr = System.arraycopy(arr, i, tmp, 0, 10)
      val maxValue = tmp.max
      for (j <- i to i + 9) {
        if (arr(j) == maxValue)
          arr(j) = 1
        else
          arr(j) = 0
      }
    }
//    //println(arr)
    val V2 = new DenseMatrix(itemCount.toInt, 10, arr)
    //输出矩阵行数即itemCount
    val m = V2.numRows - 1
    //println(m) //1682
    //println(U1.count()) //943
    //println(userCount) //944
    //println(itemCount)
    val U2 = U1.map { x =>
      //x.argmax什么意思
      var argmax = x.argmax
      var arr = x.toArray
     // println(arr.length)  //10
      for (i <- 0 to 9) {
        if (i == argmax)
          arr(i) = 1
        else
          arr(i) = 0
      }
      arr
    }.collect() //将rdd转化成数组并返回
    val n = U2.length - 1
    //println(n+1)  //943
    //println(U2(0)) //U2是矩阵
    //   U2.take(10).foreach(println)  //输出的结果什么意思
    val CBT = LearningCBT(n,m,train1,ratingEntry,U2,V2,10,10,0.1,B)
    //10行10列的二维数组
    var cBT = ofDim[Double](10,10)
    //cBT为对角矩阵
      for(i <- 0 to 9){
        for(j <- 0 to 9){
          if(j == i)
            cBT(i)(j) = CBT(j)
          else
            cBT(i)(j) = 0
        }
      }
    //目标领域利用CBT
    val text = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens100k用户重叠/tD/train")
    val train = text.map(line => line.split(",")).map(line => (line(0)toInt,line(1)toInt,line(2).toDouble)).coalesce(1)
    //定义初始化变量
    val F = 10  //特征数量
    val ITERATIONS = 100  //迭代的次数
    val LAMBDA = 0.001  //正则参数
    //该函数实现什么，返回什么
    val PQ = LearningLFM(train,F,ITERATIONS,LAMBDA)
    val P3 = PQ(0)
    val Q3 = PQ(1)
    var sum = 0.0
    //二层循环
    val predictrating = for((k1,v1) <- P3;(k2,v2) <- Q3) yield {
      var sum = 0.0
      //until与to相比不包括F
      for(k <- 0 until F){
        sum += v1(k) * cBT(k)(k) * v2(k)
      }
      //什么意思？
      (k1,k2,if((sum/100) > 5) 5.0 else(sum/100))
    }
    val predictracingfinal = predictrating.toList
    val predictratings = sc.parallelize(predictracingfinal)
    val test = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens100k用户重叠/tD/test")
    val numTest = test.count()
    val testrating = test.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val Rmse = computeRmse(predictratings,testrating,numTest)
    println("计算得出Rmse" + Rmse)
  }
}
