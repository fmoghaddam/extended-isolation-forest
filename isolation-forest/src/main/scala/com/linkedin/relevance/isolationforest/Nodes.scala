package com.linkedin.relevance.isolationforest

/** Contains the node classes used to construct isolation trees.
  */
private[isolationforest] case object Nodes {

  /** Base trait Nodes used in the isolation forest.
    */
  sealed trait Node extends Serializable {
    val subtreeDepth: Int
  }

  /** TODO
    * External nodes in an isolation tree.
    *
    * @param numInstances The number of data points that terminated in this node during training.
    */
  case class ExternalNode(numInstances: Long) extends Node {

//    require(numInstances > 0, "parameter numInstances must be >0, but given invalid value" +
//      s" ${numInstances}")

    override val subtreeDepth: Int = 0

    override def toString: String =
      s"ExternalNode(numInstances = $numInstances)"
  }

  /** TODO
    * Internal nodes in an isolation tree.
    *
    * @param leftChild      The left child node produced by splitting the data points at this node.
    * @param rightChild     The right child node by splitting the data points at this node.
    * @param slopes         The slopes for the splitter plane.
    * @param intercepts     The intercepts for the splitter plane.
    */
  case class InternalNode(
      leftChild: Node,
      rightChild: Node,
      slopes: Array[Double],
      intercepts: Array[Double]
  ) extends Node {

//    require(splitAttribute >= 0, "parameter splitAttribute must be >=0, but given invalid value" +
//      s" ${splitAttribute}")

    override val subtreeDepth: Int =
      1 + math.max(leftChild.subtreeDepth, rightChild.subtreeDepth)

//    override def toString: String =
//      s"InternalNode(splitAttribute = $splitAttribute, splitValue = $splitValue," +
//        s" leftChild = ($leftChild), rightChild = ($rightChild))"
  }
}
