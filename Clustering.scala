package com.df.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}


object Clustering {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Clustering Example")
    val sc = new SparkContext(sparkConf)

    def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))
    val data = sc.textFile(args(0)).map(parse).cache()

    // Clustering the data into 6 clusters by BisectingKMeans.
    val bkm = new BisectingKMeans().setK(6)
    val model = bkm.run(data)

    // Show the compute cost and the cluster centers
    println(s"Compute Cost: ${model.computeCost(data)}")
    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      println(s"Cluster Center ${idx}: ${center}")
    }
    // $example off$

    sc.stop()
  }
}
//bin/spark-submit --class com.df.spark.mllib.Clustering ../sparkMLlib.jar ../mllib-data/kmeans_data.txt