package main.scala

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import test.scala.LFMtest.{LearningLFM, computeRmse}


/**
  * Created by tian on 2018/10/26.
  */

//思路：1.将Rs和Rt合成一个新矩阵R
//     2.分解R=>PQ形成新的评分R
//     3.将矩阵P、Q进行聚类形成的P'和Q'相乘形成Rc
//     4.将R和Rc进行加权得出新的评分
object CMFTF {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("CMFTF")
    val sc = new SparkContext(conf)
    val sTrainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens100k用户重叠/sD/sD")
    val dTrainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens100k用户重叠/tD/train")
    val dTestData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens100k用户重叠/tD/test")
    val sData = sTrainData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val dTrain = dTrainData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val dTest = dTestData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
//    println("sData长度： " + sData.count() + " dTrain长度： " + dTrain.count())  //50000,32138




//    此部分求用户对项目的未聚类前的评分
//  将源域和目标域的训练集构成一个大的矩阵进行分解，分解R=>PQ形成新的评分R
    val sdTrain = sData.union(dTrain).distinct()
//    println("sdTrain长度： " + sdTrain.count()) //74076
    val F = 30
    val ITERATIONS = 100
    val LAMBDA = 0.001
    println("未聚类前正在训练中---")
    val PQ = LearningLFM(sdTrain,F,ITERATIONS,LAMBDA)
    val P = PQ(0)
    val Q = PQ(1)
    println("预测未聚类前的评分")
    val predictRatings1 = for((k1,v1) <- P; (k2,v2) <- Q) yield {
      var sum = 0.0
      for(k <- 0 until F){
        sum += v1(k) * v2(k)
      }
      ((k1,k2),sum)
    }
//    predictRatings1List为未聚类前用户对项目的评分列表
    val predictRatings1Array = predictRatings1.toArray
    val predictRatings1Rdd = sc.parallelize(predictRatings1Array)
//    println("未聚类的训练集评分长度" + predictRatings1.values.toList.length)
//    val predictRatings1Rdd = sc.parallelize(predictRatings1List)
    println("读入测试文件---")
//    val dTest = dTestData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    val dTestNum = dTest.count()
    println("测试集长度" + dTestNum)
    val dTestMap = dTestData.map(_.split(",")).map(line => ((line(0).toInt,line(1).toInt),line(2).toDouble)).collect()
    val dtestMapRdd = sc.parallelize(dTestMap)
    val dTrainTest = predictRatings1Rdd.join(dtestMapRdd)
    println("测试集和训练集共有的长度" + dTrainTest.collect().length)
//    dTrainTest.take(10).foreach(println)  //（（用户，项目），（预测评分，实际评分））
//    未聚类之前的（（用户，项目），预测评分）
    val userItemPre = dTrainTest.map(x => ((x._1._1,x._1._2),x._2._1))

//    val rmse = computeRmse(predictRatings1Rdd,dTest,dTestNum)
//    println("Rmse: " + rmse)  //0.92




//    此部分求聚类后形成的评分
//    P，Q是用户特征矩阵和项目特征矩阵
    println("用户项目特征矩阵聚类中---")
    val sDataC = sTrainData.map(_.split(",") match{
      case Array(user,item,rating) => Rating(user.toInt,item.toInt,rating.toDouble)
    }).cache()
    val dTrainC = dTrainData.map(_.split(",") match{
      case Array(user,item,rating) => Rating(user.toInt,item.toInt,rating.toDouble)
    })
    val sdTrainC = sDataC.union(dTrainC).distinct()
//    println("sdTrain1长度： " + sdTrain1.count())  //74076
    val model = ALS.train(sdTrainC,30,20,0.1)  //秩(特征数)，迭代次数，正则系数 返回值：MatrixFactorizationModel(rank: Int, userFeatures: RDD[(Int, Array[Double])], productFeatures: RDD[(Int, Array[Double])])


//    用户和项目特征矩阵进行聚类
//    val uVec = model.userFeatures.map{case (uId,uArr) => uArr}.map(each => Vectors.dense(each))
    val uVec = model.userFeatures.map(each => Vectors.dense(each._2))
    val iVec = model.productFeatures.map{case (iId,iArr) => iArr}.map(each => Vectors.dense(each))
    val numClusters = 35
    val numIterations = 20
    val userClusters = KMeans.train(uVec,numClusters,numIterations)  //返回值KMeansModel(clusterCenters: Array[Vector])
    val itemClusters = KMeans.train(iVec,numClusters,numIterations)

//    此处为了测试用户项目属于那个聚类中(用户id，用户所属类)
    val userToCluster = model.userFeatures.map(each => (each._1,Vectors.dense(each._2)))
      .map(each => (each._1,userClusters.predict(each._2))).collect().toMap
    val itemToCluster = model.productFeatures.map(each => (each._1,Vectors.dense(each._2)))
      .map(each => (each._1,userClusters.predict(each._2))).collect().toMap
//    itemToCluster.take(10).foreach(println)
//    println(userToCluster.values.distinct().count())
//    println(userToCluster.length,itemToCluster.length)


    val userCenter = userClusters.clusterCenters
    val itemCenter = itemClusters.clusterCenters
//    将用户聚类的center从向量通过rdd转化成数组
    val userFeature = sc.parallelize(userCenter).map(v => v.toArray).collect()
    val itemFeature = sc.parallelize(itemCenter).map(v => v.toArray).collect()
//    println(userFeature.length)  //35
//    println(itemFeature.length)  //35


//    返回 （（用户类，项目类），评分）键值对
    val clusterRatings = for(userC <- userFeature; itemC <- itemFeature) yield {
      var sum = 0.0
      for(k <- Range(0,30)) {
        sum += userC(k) * itemC(k)
      }
      ((userClusters.predict(Vectors.dense(userC)),itemClusters.predict(Vectors.dense(itemC))),sum)
    }
    val clusterRatingsMap = clusterRatings.toMap
//    clusterRatings.take(10).foreach(println)
//    println(clusterRatings.length)  //35*35=1225


//    处理测试集
    val dUser = dTestData.map(_.split(",")).map(line => line(0).toInt).collect()
    val dItem = dTestData.map(_.split(",")).map(line => line(1).toInt).collect()
    val dUserItemRating = dTestData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt)).collect()
//    println(dUser.length,dItem.length)  //7863
//    println(dUser.distinct.length)  //409
//    println(dItem.max)  //1676
//    println(dItem.distinct.length)  //1174  测试集和训练集中项目不同怎么办

//    userClusterMap和item是映射成（用户，用户类）（项目，项目类）
    var userClusterMap = Map[Int,Int]()
    for(user <- dUser){
      if(userToCluster.contains(user)){
        userClusterMap += (user -> userToCluster(user))
      }
    }
    var itemClusterMap = Map[Int,Int]()
    for(item <- dItem){
      if(itemToCluster.contains(item)) {
        itemClusterMap += (item -> itemToCluster(item))
      }
    }
//    println(userClusterMap.toList.length)  //409
//    println(itemClusterMap.toList.length)  //1170
//      userClusterMap.values.toList.foreach(println)
//    预测评分
    var predictRatings2 =
      for{(u,v) <- dUserItemRating
          if(userToCluster.contains(u) && itemToCluster.contains(v))
         }
        yield {
//      if(userToCluster.contains(u) && itemToCluster.contains(v)){
//        ((userToCluster(u),itemToCluster(v)),clusterRatingsMap((userToCluster(u),itemToCluster(v))))
//        ((u,v),clusterRatingsMap((userToCluster(u),itemToCluster(v))))
//      }
//      ((u,v),if(userToCluster.contains(u) && itemToCluster.contains(v)) clusterRatingsMap((userToCluster(u),itemToCluster(v))) else 2.5)
//      if(userToCluster.contains(u) && itemToCluster.contains(v))
        ((u,v),clusterRatingsMap((userToCluster(u),itemToCluster(v))))
        }
    println("聚类后的评分长度" + predictRatings2.toMap.values.toList.length)
    val userItemPreC = sc.parallelize(predictRatings2)
//    0.8是权重，可以随意设置
    val userItemPreFinal = userItemPre.join(userItemPreC).map(x => (x._1._1,x._1._2,1 * x._2._1 + 0 * x._2._2))
//    userItemPreFinal.take(10).foreach(println)
//    val predictRatings2Map = predictRatings2.toMap
//    predictRatings2Map.values.foreach(println)
//    println(clusterRatingsMap((3,2)))
//    计算rmse
    println("预测出的数据长度" + userItemPreFinal.collect().length)
    val Num = userItemPreFinal.collect().length
    val rmse = computeRmse(userItemPreFinal,dTest,Num)
    println("Rmse: " + rmse)

  }
}
