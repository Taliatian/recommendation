package main.scala

import main.scala.BiasSVD.learningBiasSVD
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import main.scala.Evaluation.{computeMAE,computeRmse}

/**
  * Created by tian on 2019/1/2.
  */
object ATCFtest {

  //  使用als进行矩阵分解，返回矩阵分解类型
  def alsMatrixFac(Data: RDD[String], F: Int, INTERATIONS: Int, LAMBDA: Double): MatrixFactorizationModel= {
    val DataC = Data.map(_.split(",") match {
      case Array(user,item,rating) => Rating(user.toInt,item.toInt,rating.toDouble)
    })
    val model = ALS.train(DataC,F,INTERATIONS,LAMBDA)
    model
  }


  //  使用Kmeans进行特征矩阵聚类，返回一个群级的评分  返回值出现错误，之后再调试，要返回中心点KMeansModel，KMeansModel，Array
  def kMeans(model: MatrixFactorizationModel, numCluster: Int, iterations: Int, sc: SparkContext, F: Int) = {

    //    用户和项目特征矩阵进行聚类
    val uVec = model.userFeatures.map(each => Vectors.dense(each._2))
    val iVec = model.productFeatures.map(each => Vectors.dense(each._2))
    val userClusters = KMeans.train(uVec,numCluster,iterations)
    val itemClusters = KMeans.train(iVec,numCluster,iterations)
    val userCenter = userClusters.clusterCenters
    val itemCenter = itemClusters.clusterCenters
    //    将用户聚类的center从向量通过rdd转化成数组
    val userFeature = sc.parallelize(userCenter).map(v => v.toArray).collect()
    val itemFeature = sc.parallelize(itemCenter).map(v => v.toArray).collect()
    //    返回 （（用户类，项目类），评分）键值对
    val clusterRatings = for(userC <- userFeature; itemC <- itemFeature) yield {
      var sum = 0.0
      for (k <- Range(0, F)) {
        sum += userC(k) * itemC(k)
      }
      ((userClusters.predict(Vectors.dense(userC)), itemClusters.predict(Vectors.dense(itemC))), sum)
    }
    val value = (clusterRatings,userClusters)
    value
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("CMFTF")
    val sc = new SparkContext(conf)
    val sTrainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/douban/book整理")
    val tTrainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/douban/movie整理train")
    val tTestData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/douban/movie整理test")
//    val sTrainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens1m用户重叠/source")
//    val tTrainData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens1m用户重叠/tTrain")
//    val tTestData = sc.textFile("/Users/tian/tjy/dataset/构建的数据集/movielens1m用户重叠/tTest")
    val sData = sTrainData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble)).cache()
    val tTrain = tTrainData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble)).cache()
    val tTest = tTestData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble)).cache()
    //    println("sData长度： " + sData.count() + " dTrain长度： " + dTrain.count())  //50000,32138



    //基于知识迁移
    //源领域进行矩阵分解，目标领域进行矩阵分解，用户特征迁移
    val sdTrain = sData.union(tTrain).distinct()
    //    //    println("sdTrain长度： " + sdTrain.count()) //74076
    //    val F = 30
    //    val ITERATIONS = 100
    //    val LAMBDA = 0.001
    //    println("未聚类前正在训练中---")
    //    val PQ = LearningLFM(sdTrain,F,ITERATIONS,LAMBDA)
    //    val P = PQ(0)
    //    val Q = PQ(1)
    //    println("预测未聚类前的评分")
    //    val predictRatings1 = for((k1,v1) <- P; (k2,v2) <- Q) yield {
    //      var sum = 0.0
    //      for(k <- 0 until F){
    //        sum += v1(k) * v2(k)
    //      }
    //      ((k1,k2),sum)
    //    }
    //    //    predictRatings1List为未聚类前用户对项目的评分列表
    //    val predictRatings1Array = predictRatings1.toArray
    //    val predictRatings1Rdd = sc.parallelize(predictRatings1Array)
    //    //    println("未聚类的训练集评分长度" + predictRatings1.values.toList.length)
    //    //    val predictRatings1Rdd = sc.parallelize(predictRatings1List)
    //    println("读入测试文件---")
    //    //    val dTest = dTestData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt,line(2).toDouble))
    //    val dTestNum = dTest.count()
    //    println("测试集长度" + dTestNum)
    //    val dTestMap = dTestData.map(_.split(",")).map(line => ((line(0).toInt,line(1).toInt),line(2).toDouble)).collect()
    //    val dtestMapRdd = sc.parallelize(dTestMap)
    //    val dTrainTest = predictRatings1Rdd.join(dtestMapRdd)
    //    println("测试集和训练集共有的长度" + dTrainTest.collect().length)
    //    //    dTrainTest.take(10).foreach(println)  //（（用户，项目），（预测评分，实际评分））
    //    //    未聚类之前的（（用户，项目），预测评分）
    //    val userItemPre = dTrainTest.map(x => ((x._1._1,x._1._2),x._2._1))
    //
    //    //    val rmse = computeRmse(predictRatings1Rdd,dTest,dTestNum)
    //    //    println("Rmse: " + rmse)  //0.92




    //    第一步 是将源领域的特征矩阵和目标领域的特征矩阵进行合并实现 transfer
    println("迁移源领域用户特征正在训练中---")
    var ratingEntry = sData.map(each => MatrixEntry(each._1,each._2,each._3))
    val userCount = sData.map(_._1).max
    val itemCount = sData.map(_._2).max
    var F = 25
    //    var F2 = 35
    var INTERATIONS = 60
    val LAMBDA = 0.001
    val sDataRating = sTrainData.map(_.split(",")).map(line => line(2).toDouble)
    val smu = sDataRating.sum() / sDataRating.count()
    val sAlpha = 0.01
    //    分解源领域
    val UVs = learningBiasSVD(sData,F,sAlpha,INTERATIONS,LAMBDA,smu)
    val Us = UVs._1
    val tDataRating = tTrainData.map(_.split(",")).map(line => line(2).toDouble)
    val tmu = tDataRating.sum() / tDataRating.count()
    //    分解目标领域
    val UVt = learningBiasSVD(tTrain,F,sAlpha,INTERATIONS,LAMBDA,tmu)
    val Ut = UVt._1
    val Vt = UVt._2
    val bu = UVt._3
    val bi = UVt._4
    //    println(Us.toList.length + " " + Ut.toList.length)  //用户数目
    //    var U = new Array[(Int,mutable.Buffer[Double])](Us.length)
    //    将相同用户的特征进行加权，不同的不变。
    var a = 0.5
    //    var i = 0
    for(u1 <- Us; u2 <- Ut){
      if(u1._1 == u2._1){
        for(k <- 0 until F){
          u2._2(k) = (1 - a) * u1._2(k) + a * u2._2(k)
        }
        //        i = i + 1
      }
    }
    //    val predictRatings1 = for((k1,v1) <- Ut; (k2,v2) <- Vt) yield {
    //      var sum = 0.0
    //      for(k <- 0 until F) {
    //        sum = sum + v1(k) * v2(k)
    //        //println(sum)
    //      }
    //      ((k1,k2),sum)
    //    }
//    predictRatings1为迁移阶段 用户-项目-评分
    val predictRatings1 = for ((k1,v1) <- Ut; (k2,v2) <- Vt) yield {
      var ret = bu(k1) + bi(k2) + tmu
      for (k <- 0 until F) {
        ret += v1(k) * v2(k)
      }
      ((k1, k2), ret)
    }
    val predictRatings1Array = predictRatings1.toArray
    val predictRatingRdd = sc.parallelize(predictRatings1Array)
    val tTestMap = tTestData.map(_.split(",")).map(line => ((line(0).toInt,line(1).toInt),line(2).toDouble)).collect()
    val dtestMapRdd = sc.parallelize(tTestMap)
    val dTrainTest = predictRatingRdd.join(dtestMapRdd)
//    userItemPre为用户-项目-评分  知识迁移阶段
    val userItemPre = dTrainTest.map(x => ((x._1._1,x._1._2),x._2._1))





    //    源领域进行分解,可以使用LFM，也可以使用ALS等矩阵分解算法
    println("源领域正在分解中---")
    var sF = 30
    var sINTERATIONS = 20
    val sLAMBDA = 0.001
    //    val UVs = LearningLFM(sData,sF,sINTERATIONS,sLAMBDA)
    //    val Us = UVs(0)
    //    val Vs = UVs(1)
    val modelSource = alsMatrixFac(sTrainData,sF,sINTERATIONS,sLAMBDA)
    //    modelSource.userFeatures.take(2).foreach(println)
    //    （rank: Int, userFeatures: RDD[(Int, Array[Double])], productFeatures: RDD[(Int, Array[Double])]）





    //    对源领域分解的特征矩阵进行聚类，然后形成新的聚类后的用户群和项目群的评分矩阵
    println("源领域正在聚类中---")
    val sourceNumCluster = 150
    val sourceIterations = 20
    //    获得聚类之后的评分 ((0,0),2.227598260900311)
    //    返回值为（用户类项目类评分，中心点）
    val sourceClusterRating = kMeans(modelSource,sourceNumCluster,sourceIterations,sc,sF)
    //    val uVecSource = modelSource.userFeatures.map(each => Vectors.dense(each._2))
    //    val iVecSource = modelSource.productFeatures.map(each => Vectors.dense(each._2))
    //    val userClustersSource = KMeans.train(uVecSource,sourceNumCluster,sourceIterations)
    //    val itemClustersSource = KMeans.train(iVecSource,sourceNumCluster,sourceIterations)
    //    val userCenterSource = userClustersSource.clusterCenters
    //    val itemCenterSource = itemClustersSource.clusterCenters
    //    //    将用户聚类的center从向量通过rdd转化成数组
    //    val userFeatureSource = sc.parallelize(userCenterSource).map(v => v.toArray).collect()
    //    val itemFeatureSource = sc.parallelize(itemCenterSource).map(v => v.toArray).collect()
    //    //    返回 （（用户类，项目类），评分）键值对
    //    val sourceClusterRating = for(userC <- userFeatureSource; itemC <- itemFeatureSource) yield {
    //      var sum = 0.0
    //      for (k <- Range(0, sF)) {
    //        sum += userC(k) * itemC(k)
    //      }
    //      ((userClustersSource.predict(Vectors.dense(userC)), itemClustersSource.predict(Vectors.dense(itemC))), sum)
    //    }
    //    println("源领域聚类后的评分： ")
    //    sourceClusterRating.foreach(println)

    println("源领域聚类后的评分个数： " + sourceClusterRating._1.length)
    val sourceClusterRatingsMap = sourceClusterRating._1.toMap





    //    目标领域进行分解
    println("目标领域正在分解中---")
    var tF = 30
    var tINTERATIONS = 10
    val tLAMBDA = 0.01
    val modelTarget = alsMatrixFac(tTrainData,sF,sINTERATIONS,sLAMBDA)
    //    val UVt = LearningLFM(dTrain,tF,tINTERATIONS,tLAMBDA)
    //    val Ut = UVt(0)
    //    val Vt = UVt(1)




    //    目标领域进行聚类
    println("目标领域正在聚类中---")
    val targetNumCluster = 75
    val targetIterations = 20
    //    获得聚类之后的评分 ((0,0),2.227598260900311)
    val targetClusterRating = kMeans(modelTarget,targetNumCluster,targetIterations,sc,tF)
    //    val uVecTarget = modelTarget.userFeatures.map(each => Vectors.dense(each._2))
    //    val iVecTarget = modelTarget.productFeatures.map(each => Vectors.dense(each._2))
    //    val userClustersTarget = KMeans.train(uVecTarget,targetNumCluster,targetIterations)
    //    val itemClustersTarget = KMeans.train(iVecTarget,targetNumCluster,targetIterations)
    //    val userCenterTarget = userClustersTarget.clusterCenters
    //    val itemCenterTarget = itemClustersTarget.clusterCenters
    //    //    将用户聚类的center从向量通过rdd转化成数组
    //    val userFeatureTarget = sc.parallelize(userCenterTarget).map(v => v.toArray).collect()
    //    val itemFeatureTarget = sc.parallelize(itemCenterTarget).map(v => v.toArray).collect()
    //    //    返回 （（用户类，项目类），评分）键值对
    //    val targetClusterRating = for(userC <- userFeatureTarget; itemC <- itemFeatureTarget) yield {
    //      var sum = 0.0
    //      for (k <- Range(0, tF)) {
    //        sum += userC(k) * itemC(k)
    //      }
    //      ((userClustersTarget.predict(Vectors.dense(userC)), itemClustersTarget.predict(Vectors.dense(itemC))), sum)
    //    }
    //    println("目标领域聚类后的评分： ")
    //    targetClusterRating.foreach(println)
    println("目标领域聚类后的评分个数： " + targetClusterRating._1.length)
    val targetClusterRatingsMap = targetClusterRating._1.toMap




    //    聚类后的源领域和目标领域进行合并,不能直接合并，需要对目标领域的聚类进行处理然后合并成大矩阵
    val sourceClusterRatingRdd1 = sc.parallelize(sourceClusterRating._1)
    val sourceClusterRatingRdd = sourceClusterRatingRdd1.map(each => (each._1._1,each._1._2,each._2))
    println("源领域： ")
    sourceClusterRatingRdd.take(5).foreach(println)
    val targetClusterRatingRdd1 = sc.parallelize(targetClusterRating._1)
    val targetClusterRatingRdd = targetClusterRatingRdd1.map(each => (each._1._1,each._1._2,each._2))
    //    第一个+表示加源领域用户聚类的个数，第二个+表示加源领域项目聚类的个数，
    //    因为前面聚类函数中传参中没有分，所以一样，后续改的时候可以分别传参
    val targetClusterRatingRddPlus = targetClusterRatingRdd.map(each => (each._1 + sourceNumCluster,each._2 + sourceNumCluster,each._3))
    val sourceTarget = sourceClusterRatingRdd.union(targetClusterRatingRddPlus)
    println("目标领域")
    targetClusterRatingRddPlus.take(5).foreach(println)
    //    println("合并后的矩阵用户类最大值" + sourceTarget.map(each => each._1).max())
    //    sourceTarget.take(100).foreach(println)
    //    println("重复前：" + sourceTarget.count())
    //    println("重复后：" + sourceTarget.distinct().count())




    //    对聚类后的评分模式进行矩阵分解    可以测试用LFM效果怎样
    val clusterF = 30
    val clusterIteration = 70
    val clusterLambda = 0.001
    println("聚类后的评分矩阵大小为： " +  (sourceNumCluster + targetNumCluster) + "*" + (sourceNumCluster + targetNumCluster))
    val muCluster =sourceTarget.map(_._3).reduce(_ + _) / sourceTarget.count()
    println("聚类后的评分值：" + muCluster)
    val alpha = 0.01
    val UVCluster = learningBiasSVD(sourceTarget,clusterF,alpha,clusterIteration,clusterLambda,muCluster)
    val uCluster = UVCluster._1
    val vCluster = UVCluster._2
    val buCluster = UVCluster._3
    val biCluster = UVCluster._4
    //    预测评分
    println("开始预测评分---")
    val predictCluster = for ((k1,v1) <- uCluster; (k2,v2) <- vCluster) yield {
      var ret = buCluster(k1) + biCluster(k2) + muCluster
      for (k <- 0 until clusterF) {
        ret += v1(k) * v2(k)
      }
      (k1, k2, ret)
    }
    //    predictCluster.foreach(println)
    println("聚类后的评分模式分解后评分预测结束---")
    println("合并分解聚类后评分个数： " + predictCluster.toList.length)




    //    将聚类后的评分模式进行评分填充，此步重点在于找到类中的用户和类中项目进行平均分填充



    //    解决方案二：对于训练集中每个用户和项目寻找所属类别，然后进行预测评分   此处分析错误

    //    解决方案一：只需要找到测试集中用户和项目所属类别，然后进行预测评分 ，供后续计算rmse
    println("源领域评分个数： " + sData.count())
    println("目标域训练集评分个数： " + tTrain.count())
    val stTrains = sData.union(tTrain)
    println("合并后的训练集评分个数： " + stTrains.count())



    //    计算用户，项目所属类
    val clusterRatingsMap = predictCluster.map(each => ((each._1,each._2),each._3)).toMap


    //    (用户id，用户所属类)
    //    计算源用户所属类
    val userToClusterSource = modelSource.userFeatures.map(each => (each._1,Vectors.dense(each._2)))
      .map(each => (each._1,sourceClusterRating._2.predict(each._2)))
    //    计算源项目所属类
    val itemToClusterSource = modelSource.productFeatures.map(each => (each._1,Vectors.dense(each._2)))
      .map(each => (each._1,sourceClusterRating._2.predict(each._2)))
    //    计算目标用户所属类
    val userToClusterTarget = modelTarget.userFeatures.map(each => (each._1,Vectors.dense(each._2)))
      .map(each => (each._1,targetClusterRating._2.predict(each._2))).map(each => (each._1,each._2 + sourceNumCluster))
    //    计算目标项目所属类
    val itemToClusterTarget = modelTarget.productFeatures.map(each => (each._1,Vectors.dense(each._2)))
      .map(each => (each._1,targetClusterRating._2.predict(each._2))).map(each => (each._1,each._2 + sourceNumCluster))
    val userToCluster = userToClusterSource.union(userToClusterTarget).collect().toMap
    val itemToCluster = itemToClusterSource.union(itemToClusterTarget).collect().toMap

    //    遍历整合后的大矩阵用户和项目
    //         println("0类对0类的评分" + clusterRatingsMap((0,0)))

    //    一下三行无用
    val trainUser = stTrains.map(each => each._1)
    val trainItem = stTrains.map(each => each._2)
    val trainUserItem = stTrains.map(each => (each._1,each._2)).collect()

    val tUser = tTestData.map(_.split(",")).map(line => line(0).toInt).collect()
    val tItem = tTestData.map(_.split(",")).map(line => line(1).toInt).collect()
    val tUserItemRating = tTestData.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt)).collect()
//    prediction为用户-项目-聚类形成的评分
    val predictRatings2 =
      for{(u,v) <- tUserItemRating
          if(userToCluster.contains(u) && itemToCluster.contains(v))
      } yield {
        var clusterRating = 0.0
        if(clusterRatingsMap((userToCluster(u),itemToCluster(v))) < 0){
          clusterRating = 1
        }
        else if(clusterRatingsMap((userToCluster(u),itemToCluster(v))) > 5){
          clusterRating = 5
        }
        else{
          clusterRating = clusterRatingsMap((userToCluster(u),itemToCluster(v)))
        }
        ((u,v),clusterRating)
      }
    val userItemPreC = sc.parallelize(predictRatings2)
    val userItemPreFinal = userItemPre.join(userItemPreC).map(x => (x._1._1,x._1._2,0.5 * x._2._1 + 0.5 * (x._2._2 + 1.5)))
    val userItemTest = userItemPre.join(userItemPreC).map(x => ((x._1._1,x._1._2),(x._2._1,x._2._2)))
    val tTestMap1 = tTestData.map(_.split(",")).map(line => ((line(0).toInt,line(1).toInt),(line(2).toDouble,0.0)))
    val userItemTest1 = userItemTest.join(tTestMap1)
    userItemTest1.take(10).foreach(println)
    println("正在计算rmse值---")
    val rmse = computeRmse(userItemPreFinal,tTest)
    println("rmse： " + rmse)
  }
}
