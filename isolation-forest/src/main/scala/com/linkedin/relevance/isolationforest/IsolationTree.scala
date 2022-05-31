package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Nodes.{
  ExternalNode,
  InternalNode,
  Node
}
import com.linkedin.relevance.isolationforest.Utils.DataPoint
import org.apache.spark.internal.Logging

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer
import scala.util.Random

/** A trained isolation tree.
  *
  * @param node The root node of the isolation tree model.
  */
private[isolationforest] class IsolationTree(val node: Node)
    extends Serializable {

  import IsolationTree._

  /** Returns the path length from the root node of this isolation tree to the node in the tree that
    * contains a particular data point.
    *
    * @param dataInstance The feature array for a single data instance.
    * @return The path length to the instance.
    */
  private[isolationforest] def calculatePathLength(
      dataInstance: DataPoint
  ): Float =
    pathLength(dataInstance, node)
}

/** Companion object used to train the IsolationTree class.
  */
private[isolationforest] case object IsolationTree extends Logging {

  /** Fits a single isolation tree to the input data.
    *
    * @param data The 2D array containing the feature values (columns) for the data instances (rows)
    *             used to train this particular isolation tree.
    * @param randomSeed The random seed used to generate this tree.
    * @param featureIndices Array containing the feature column indices used for training this
    *                       particular tree.
    * @return A trained isolation tree object.
    */
  def fit(
      data: Array[DataPoint],
      randomSeed: Long,
      featureIndices: Array[Int]
  ): IsolationTree = {

    logInfo(
      s"Fitting isolation tree with random seed ${randomSeed} on" +
        s" ${featureIndices.seq.toString} features (indices) using ${data.length} data points."
    )

    def log2(x: Double): Double = math.log10(x) / math.log10(2.0)
    val heightLimit = math.ceil(log2(data.length.toDouble)).toInt

    new IsolationTree(
      generateIsolationTree(
        data,
        heightLimit,
        new Random(randomSeed),
        featureIndices
      )
    )
  }

  /** TODO Fix me for better performance
    * @param data
    * @param size
    * @param randomState
    * @return
    */
  def getIntercepts(
      data: Array[DataPoint],
      size: Int,
      randomState: Random
  ): Array[Double] = {
    val result: Array[Double] = Array.fill(size)(0)
    if (data.nonEmpty) {
      for (i <- 0 until size) {
        val featureValues = data.map(x => x.features(i))
        val minFeatureValue = featureValues.min.toDouble
        val maxFeatureValue = featureValues.max.toDouble
        result(i) =
          ((maxFeatureValue - minFeatureValue) * randomState.nextDouble
            + minFeatureValue)
      }
    }
    result
  }

  /** TODO better performance
    * @param x
    * @param slopes
    * @param intercepts
    * @return
    */
  def check(
      x: DataPoint,
      slopes: Array[Double],
      intercepts: Array[Double]
  ): Double = {
    val sub: Array[Double] = Array.fill(x.features.length) {
      0.0
    }
    for (i <- 0 until x.features.length) {
      sub(i) = x.features(i) - intercepts(i)
    }
    var sum = 0.0
    for (i <- 0 until x.features.length) {
      sum = sum + sub(i) * slopes(i)
    }
    sum
  }

  /** Generates an isolation tree. It encloses the generateIsolationTreeInternal() method to hide the
    * currentTreeHeight parameter.
    *
    * @param data Feature data used to generate the isolation tree.
    * @param heightLimit The tree height at which the algorithm terminates.
    * @param randomState The random state object.
    * @param featureIndices Array containing the feature column indices used for training this
    *                       particular tree.
    * @return The root node of the isolation tree.
    */
  def generateIsolationTree(
      data: Array[DataPoint],
      heightLimit: Int,
      randomState: Random,
      featureIndices: Array[Int]
  ): Node = {

    /** This is a recursive method that generates an isolation tree. It is an implementation of the
      * iTree(X,e,l) algorithm in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
      *
      * @param data Feature data used to generate the isolation tree.
      * @param currentTreeHeight Height of the current tree. Initialize this to 0 for a new tree.
      * @param heightLimit The tree height at which the algorithm terminates.
      * @param randomState The random state object.
      * @param featureIndices Array containing the feature column indices used for training this
      *                       particular tree.
      * @return The root node of the isolation tree.
      */
    def generateIsolationTreeInternal(
        data: Array[DataPoint],
        currentTreeHeight: Int,
        heightLimit: Int,
        randomState: Random,
        featureIndices: Array[Int]
    ): Node = {

      /** Randomly selects a feature and feature value to split upon.
        *
        * @param data The data at the particular node in question.
        * @return Tuple containing the feature index and the split value. Feature index is -1 if no
        *         features could be split.
        */
      def getFeatureToSplit(
          data: Array[DataPoint]
      ): (Array[Double], Array[Double]) = {
        val availableFeatures = featureIndices.to[ListBuffer]
        val slopes =
          Array.fill(availableFeatures.length)(randomState.nextDouble())
        val intercepts =
          getIntercepts(data, availableFeatures.length, randomState)
        (slopes, intercepts)
      }

      val (slopes, intercepts) = getFeatureToSplit(data)
      val numInstances = data.length

      if (currentTreeHeight >= heightLimit || numInstances <= 1)
        ExternalNode(numInstances)
      else {
        val dataLeft = data.filter(x => check(x, slopes, intercepts) < 0)
        val dataRight = data.filter(x => check(x, slopes, intercepts) >= 0)

        InternalNode(
          generateIsolationTreeInternal(
            dataLeft,
            currentTreeHeight + 1,
            heightLimit,
            randomState,
            featureIndices
          ),
          generateIsolationTreeInternal(
            dataRight,
            currentTreeHeight + 1,
            heightLimit,
            randomState,
            featureIndices
          ),
          slopes,
          intercepts
        )
      }
    }

    generateIsolationTreeInternal(
      data,
      0,
      heightLimit,
      randomState,
      featureIndices
    )
  }

  /** Returns the path length from the root node of an isolation tree to the node in the tree that
    * contains a particular data point.
    *
    * @param dataInstance      A single data point for scoring.
    * @param node              The root node of the tree used to calculate the path length.
    * @return The path length to the instance.
    */
  def pathLength(dataInstance: DataPoint, node: Node): Float = {

    /** This recursive method returns the path length from a node of an isolation tree to the node
      * in the tree that contains a particular data point. The returned path length includes an
      * additional component dependent upon how many training data points ended up in this node. This
      * is the PathLength(x,T,e) algorithm in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
      *
      * @param dataInstance      A single data point for scoring.
      * @param node              The root node of the tree used to calculate the path length.
      * @param currentPathLength The path length to the current node.
      * @return The path length to the instance.
      */
    @tailrec
    def pathLengthInternal(
        dataInstance: DataPoint,
        node: Node,
        currentPathLength: Float
    ): Float = {

      node match {
        case externalNode: ExternalNode =>
          currentPathLength + Utils.avgPathLength(externalNode.numInstances)
        case internalNode: InternalNode =>
          val slopes = internalNode.slopes
          val intercepts = internalNode.intercepts
          if (check(dataInstance, slopes, intercepts) < 0) {
            pathLengthInternal(
              dataInstance,
              internalNode.leftChild,
              currentPathLength + 1
            )
          } else {
            pathLengthInternal(
              dataInstance,
              internalNode.rightChild,
              currentPathLength + 1
            )
          }
      }
    }

    pathLengthInternal(dataInstance, node, 0)
  }
}
