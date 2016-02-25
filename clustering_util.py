__author__ = "Michael Kern"
__email__ = 'kernm@in.tum.de'

########################################################################################################################

import random
import numpy as np
from scipy.stats import pearsonr, spearmanr

"""
http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
--> good explanation to create weighted choices / random numbers
"""
def weightedChoice(weights):

    # compute sum of all weights
    sumTotal = sum(weights)
    # compute a random with range[0, sumTotal]
    rnd = random.random() * sumTotal

    for index, weight in enumerate(weights):
        # subtract current weight from random to find current index
        rnd -= weight
        if rnd < 0:
            return index

    # 20% faster if weights are sorted in descending order

########################################################################################################################

"""
Implementation of an binary tree for hierarchical clustering
"""

"""
Node of the tree containing information about it's id in data, children and the value
"""
class BinaryNode:
    def __init__(self, value, id, size, leftChild, rightChild):
        self.value = value
        self.left = leftChild
        self.right = rightChild
        self.size = size
        self.id = id
        self.parent = None
        self.indices = [id]

        # create json info on the fly
        self.json = {"id": self.id, "size": self.size, "value": self.value}
        if leftChild is not None and rightChild is not None:
            self.json["value"] = np.mean(self.value)
            self.json["children"] = [leftChild.json, rightChild.json]
            self.indices = [] + rightChild.indices + leftChild.indices
            self.json["indices"] = self.indices

    def isLeave(self):
        return self.left is None and self.right is None

########################################################################################################################

"""
Implementation of an hierarchical binary tree
"""
class BinaryTree:
    # this tree must not be empty and must have at least two children (leaves)
    def __init__(self, leftNode, rightNode, newID, newValue):
        self.__createNewRoot(leftNode, rightNode, newID, newValue)

    def addNode(self, newNode, newID, newValue):
        self.__createNewRoot(self.root, newNode, newID, newValue)
        return self

    def merge(self, tree, newID, newValue):
        self.__createNewRoot(self.root, tree.root, newID, newValue)
        return self

    def jsonify(self):
        import json
        return json.dumps(self.root.json)
        # return self.root.json
        # return self.__traverseJson(self.root)

    def cutTreeByClusters(self, k):
        queue = [self.root]

        while len(queue) < k:
            node = queue.pop(0)
            queue.append(node.left)
            queue.append(node.right)

            def keyFunc(x):
                if x.isLeave():
                    return 0
                else:
                    return -x.value

            queue.sort(key=keyFunc)

        clusters = []

        for node in queue:
            clusters.append(node.indices)

        return clusters



    def __traverseJson(self, node):
        json = {"id": node.id, "size": node.size, "value": node.value}
        if node.left is None and node.right is None:
            return json
        else:
            json["children"] = [] + [self.__traverseJson(node.left)] + [self.__traverseJson(node.right)]

        return json

    def getLeaves(self):
        return self.root.indices
        # return self.__traverseIDs(self.root)

    def __traverseIDs(self, node):

        if node.left is None and node.right is None:
            return [node.id]
        else:
            return [] + self.__traverseIDs(node.right) + self.__traverseIDs(node.left)

    def __createNewRoot(self, leftNode, rightNode, newID, newValue):
            newSize = leftNode.size + rightNode.size
            self.root = BinaryNode(newValue, newID, newSize, leftNode, rightNode)
            leftNode.parent = rightNode.parent = self.root

########################################################################################################################

def squaredEuclideanDistance(matrix, vector):
    """
    Computes the euclidean distance between a vector and the rows of a matrix in parallel.
    :param matrix: array of observations or clusters
    :param vector: cluster centroid or observation
    :return:
    """

    # compute distance between values in matrix and the vector
    distMat = matrix - vector
    numValues = len(matrix)
    distances = np.array([0.0 for _ in range(numValues)], dtype=np.float)

    for ii in range(numValues):
        distance = distMat[ii]
        # always try to use np.dot when computing euclidean distance
        # it's way faster than ** 2 and sum(..., axis=1)
        distances[ii] = np.dot(distance, distance)

    return distances

def euclideanDistance(matrix, vector):
    return np.sqrt(euclideanDistance(matrix, vector))

# ----------------------------------------------------------------------------------------------------------------------

def squaredEuclideanDistanceMatrix(matrix, n):
    """
    Compute the euclidean distance matrix required for the algorithm
    :param matrix:
    :param n:
    :return:
    """

    distMat = np.zeros((n, n))

    # use Gram matrix and compute distances without inner products | FASTER than row-by-row method
    "Gramiam matrix to compute dot products of each pair of elements: "
    "<https://en.wikipedia.org/wiki/Gramian_matrix>"
    gramMat = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii, n):
            gramMat[ii, jj] = np.dot(matrix[ii], matrix[jj])

    # # ! This is slower than computing dot products of rows manually in python
    # # ! And we only require the upper triangle matrix of the Gram matrix
    # gramMat = np.dot(self.__obs, self.__obs.T)

    # make use of formula |a - b|^2 = a^2 - 2ab + b^2
    for ii in range(n):
        # self.__d[ii, ii] = self.__maxValue
        jj = np.arange(ii + 1, n)
        distMat[ii, jj] = gramMat[ii, ii] - 2 * gramMat[ii, jj] + gramMat[jj, jj]
        distMat[jj, ii] = distMat[ii, jj]

    # # take square root of distances to compute real euclidean distance
    # distMat = np.sqrt(distMat)

    "alternative version --> use scipy's fast euclidean distance implementation: FASTEST"
    # distMat = spt.distance.pdist(self.__obs, 'euclidean')
    # self.__d = spt.distance.squareform(distMat)
    # print(distMat)

    return distMat


def euclideanDistanceMatrix(matrix, n):
    return np.sqrt(squaredEuclideanDistanceMatrix(matrix, n))

# ----------------------------------------------------------------------------------------------------------------------

def norm1Distance(matrix, vector):
    """
    Computes the norm-1 distance between a vector and the rows of a matrix in parallel.
    :param matrix: array of observations or clusters
    :param vector: cluster centroid or observation
    :return:
    """
    distMat = np.abs(matrix - vector)
    numValues = len(vector)

    distances = np.sum(distMat, axis=1) / numValues
    return distances

# ----------------------------------------------------------------------------------------------------------------------

"""
Computes the mahalanobis distance between a vector and the rows of a matrix in parallel.
"""
# TODO! implement method or provide generic distance function
def mahalanobisDistance(matrix, vector):
    pass

# ----------------------------------------------------------------------------------------------------------------------

def pearsonCorrelationMatrix(matrix, n):
    """

    :param matrix:
    :param n:
    :return:
    """
    distMat = np.zeros((n, n))

    for ii in range(n):
        rowI = matrix[ii]
        for jj in range(ii + 1, n):
            rowJ = matrix[jj]
            pcc, _ = pearsonr(rowI, rowJ)
            # TODO! other possibilites like 1 - abs(corr) | sqrt(1 - corr ** 2) | (1 - corr) / 2
            corr = 1 - pcc

            distMat[ii, jj] = corr
            distMat[jj, ii] = corr

    return distMat

# ----------------------------------------------------------------------------------------------------------------------

def spearmanCorrelationMatrix(matrix, n):
    """

    :param matrix:
    :param n:
    :return:
    """
    distMat = np.zeros((n, n))

    for ii in range(n):
        rowI = matrix[ii]
        for jj in range(ii + 1, n):
            rowJ = matrix[jj]
            pcc, _ = spearmanr(rowI, rowJ)
            # TODO! other possibilites like 1 - abs(corr) | sqrt(1 - corr ** 2) | (1 - corr) / 2
            corr = 1 - pcc

            distMat[ii, jj] = corr
            distMat[jj, ii] = corr

    return distMat

########################################################################################################################
# utility functions to compute distances between rows and cluster centroids

def computeClusterInternDistances(matrix, labels, sorted=True):
    """
    Computes the distances of each element in one cluster to the cluster's centroid. Returns distance values and labels
    sorted in ascending order.
    :param matrix:
    :param labels:
    :return: labels / indices of elements corresponding to distance array, distance values of cluster
    """
    clusterLabels = np.array(labels)
    subMatrix = matrix[clusterLabels]
    # compute centroid of cluster along column (as we want to average each gene separately)
    centroid = np.mean(subMatrix, axis=0)

    # compute distances to centroid
    dists = np.sqrt(squaredEuclideanDistance(subMatrix, centroid))

    if sorted:
        # sort values
        indices = range(len(dists))
        indices.sort(key=dists.__getitem__)
        dists.sort()

        # write back to our arrays
        distLabels = clusterLabels[indices].tolist()
        distValues = dists.tolist()
    else:
        distLabels = clusterLabels.tolist()
        distValues = dists.tolist()

    return distLabels, distValues

# ----------------------------------------------------------------------------------------------------------------------

def computeClusterExternDistances(matrix, labels, outerLabels):
    """
    Compute the distances of patients in one cluster to the centroids of all other clusters.
    :param matrix:
    :param labels:
    :param outerLabels:
    :return:
    """
    externDists = []
    internSubMatrix = matrix[labels]

    for externLabels in outerLabels:
        # compute centroid of external cluster
        subMatrix = matrix[externLabels]
        centroid = np.mean(subMatrix, axis=0)

        dists = np.sqrt(squaredEuclideanDistance(internSubMatrix, centroid))
        externDists.append(dists.tolist())

    return externDists

########################################################################################################################
