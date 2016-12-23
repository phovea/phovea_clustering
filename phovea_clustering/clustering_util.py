__author__ = "Michael Kern"
__email__ = 'kernm@in.tum.de'

########################################################################################################################

import random
import numpy as np

# use scipy to compute different distance matrices
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats

"""
http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
--> good explanation to create weighted choices / random numbers
"""


def weighted_choice(weights):
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
    self.json = {"id": self.id, "size": self.size, "value": self.value, "indices": [id]}
    if leftChild is not None and rightChild is not None:
      # self.json["value"] = np.mean(self.value)
      self.json["children"] = [rightChild.json, leftChild.json]
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

  # ------------------------------------------------------------------------------------------------------------------

  def addNode(self, newNode, newID, newValue):
    self.__createNewRoot(self.root, newNode, newID, newValue)
    return self

  # ------------------------------------------------------------------------------------------------------------------

  def merge(self, tree, newID, newValue):
    self.__createNewRoot(self.root, tree.root, newID, newValue)
    return self

  # ------------------------------------------------------------------------------------------------------------------

  def jsonify(self):
    import json
    return json.dumps(self.root.json)
    # return self.root.json
    # return self.__traverseJson(self.root)

  # ------------------------------------------------------------------------------------------------------------------

  def json(self):
    return self.root.json

  # ------------------------------------------------------------------------------------------------------------------

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

  # ------------------------------------------------------------------------------------------------------------------

  def __traverseJson(self, node):
    json = {"id": node.id, "size": node.size, "value": node.value}
    if node.left is None and node.right is None:
      return json
    else:
      json["children"] = [] + [self.__traverseJson(node.left)] + [self.__traverseJson(node.right)]

    return json

  # ------------------------------------------------------------------------------------------------------------------

  def getLeaves(self):
    return self.root.indices
    # return self.__traverseIDs(self.root)

  # ------------------------------------------------------------------------------------------------------------------

  def __traverseIDs(self, node):

    if node.left is None and node.right is None:
      return [node.id]
    else:
      return [] + self.__traverseIDs(node.right) + self.__traverseIDs(node.left)

  # ------------------------------------------------------------------------------------------------------------------

  def __createNewRoot(self, leftNode, rightNode, newID, newValue):
    newSize = leftNode.size + rightNode.size
    self.root = BinaryNode(newValue, newID, newSize, leftNode, rightNode)
    leftNode.parent = rightNode.parent = self.root

    # ------------------------------------------------------------------------------------------------------------------


def cutJsonTreeByClusters(jsonData, k):
  # import json
  # tree = json.loads(jsonData)
  queue = [jsonData]

  while len(queue) < k:
    node = queue.pop(0)
    queue.append(node['children'][0])
    queue.append(node['children'][1])

    def keyFunc(x):
      if 'children' not in x:
        return 0
      else:
        return -x['value']

    queue.sort(key=keyFunc)

  clusters = []

  for node in queue:
    clusters.append(node['indices'])

  return clusters


########################################################################################################################

def euclideanDistance(matrix, vector, squared=False):
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

  if squared:
    return distances
  else:
    return np.sqrt(distances)


# ----------------------------------------------------------------------------------------------------------------------

def correlationDistance(matrix, vector, method):
  """

  :param matrix:
  :param vector:
  :return:
  """

  numValues = len(matrix)
  distances = np.array([0.0 for _ in range(numValues)], dtype=np.float)

  for ii in range(numValues):
    value = matrix[ii]

    if method == 'pearson':
      distances[ii], _ = stats.pearsonr(value, vector)
    elif method == 'spearman':
      distances[ii], _ = stats.spearmanr(value, vector)
    elif method == 'kendall':
      distances[ii], _ = stats.kendalltau(value, vector)
    else:
      raise AttributeError

  return distances


# ----------------------------------------------------------------------------------------------------------------------

from scipy.spatial.distance import cdist


def similarityMeasurement(matrix, vector, method='euclidean'):
  if method == 'euclidean':
    return euclideanDistance(matrix, vector)

  if method == 'sqeuclidean':
    return euclideanDistance(matrix, vector, True)

  spatialMethods = ['cityblock', 'chebyshev', 'canberra', 'correlation', 'hamming', 'mahalanobis', ]

  if method in spatialMethods:
    return np.nan_to_num(cdist(matrix, np.atleast_2d(vector), method).flatten())

  corrMethods = ['spearman', 'pearson', 'kendall']

  if method in corrMethods:
    return correlationDistance(matrix, vector, method)

  raise AttributeError


# ----------------------------------------------------------------------------------------------------------------------

def euclideanDistanceMatrix(matrix, squared=False):
  """
  Compute the euclidean distance matrix required for the algorithm
  :param matrix:
  :param n:
  :return:
  """

  n = np.shape(matrix)[0]
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

  if squared:
    return distMat
  else:
    return np.sqrt(distMat)


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

def pearsonCorrelationMatrix(matrix):
  """

  :param matrix:
  :param n:
  :return:
  """
  # TODO! other possibilites like 1 - abs(corr) | sqrt(1 - corr ** 2) | (1 - corr) / 2
  distMat = 1 - np.corrcoef(matrix)

  return distMat


# ----------------------------------------------------------------------------------------------------------------------

def statsCorrelationMatrix(matrix, method):
  if method == 'pearson':
    return pearsonCorrelationMatrix(matrix)

  n = np.shape(matrix)[0]
  distMat = np.zeros((n, n))

  for ii in range(n):
    rowI = matrix[ii]
    for jj in range(ii + 1, n):
      rowJ = matrix[jj]
      corr = 0

      if method == 'spearman':
        corr, _ = stats.spearmanr(rowI, rowJ)

      if method == 'kendall':
        corr, _ = stats.kendalltau(rowI, rowJ)

      # TODO! other possibilites like 1 - abs(corr) | sqrt(1 - corr ** 2) | (1 - corr) / 2
      corr = 1 - corr

      distMat[ii, jj] = corr
      distMat[jj, ii] = corr

  return distMat


# ----------------------------------------------------------------------------------------------------------------------

def similarityMeasurementMatrix(matrix, method):
  """
  Generic function to determine the similarity measurement for clustering
  :param matrix:
  :param method:
  :return:
  """
  if method == 'euclidean':
    return euclideanDistanceMatrix(matrix)
    # return squareform(pdist(matrix, method))

  if method == 'sqeuclidean':
    return euclideanDistanceMatrix(matrix, True)
    # return squareform(pdist(matrix, method))

  spatialMethods = ['cityblock', 'chebyshev', 'canberra', 'correlation', 'hamming', 'mahalanobis']

  if method in spatialMethods:
    return squareform(np.nan_to_num(pdist(matrix, method)))

  corrMethods = ['spearman', 'pearson', 'kendall']

  if method in corrMethods:
    return statsCorrelationMatrix(matrix, method)

  raise AttributeError


########################################################################################################################
# utility functions to compute distances between rows and cluster centroids

def computeClusterInternDistances(matrix, labels, sorted=True, metric='euclidean'):
  """
  Computes the distances of each element in one cluster to the cluster's centroid. Returns distance values and labels
  sorted in ascending order.
  :param matrix:
  :param labels:
  :return: labels / indices of elements corresponding to distance array, distance values of cluster
  """
  clusterLabels = np.array(labels)
  if len(clusterLabels) == 0:
    return [], []

  subMatrix = matrix[clusterLabels]
  # compute centroid of cluster along column (as we want to average each gene separately)
  centroid = np.mean(subMatrix, axis=0)

  # compute distances to centroid
  dists = similarityMeasurement(subMatrix, centroid, metric)

  if sorted == 'true':
    # sort values
    indices = range(len(dists))
    indices.sort(key=dists.__getitem__)
    dists.sort()

    # reverse order if correlation coefficient is used
    # (1 means perfect correlation while -1 denotes opposite correlation)
    corrMetrics = ['pearson', 'spearman', 'kendall']
    if metric in corrMetrics:
      indices.reverse()
      dists = dists[::-1]

    # write back to our arrays
    distLabels = clusterLabels[indices].tolist()
    distValues = dists.tolist()
  else:
    distLabels = clusterLabels.tolist()
    distValues = dists.tolist()

  return distLabels, distValues


# ----------------------------------------------------------------------------------------------------------------------

def computeClusterExternDistances(matrix, labels, outerLabels, metric='euclidean'):
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

    if len(externLabels) == 0:
      externDists.append([])

    # compute centroid of external cluster
    subMatrix = matrix[externLabels]
    centroid = np.mean(subMatrix, axis=0)

    dists = similarityMeasurement(internSubMatrix, centroid, metric)
    externDists.append(dists.tolist())

  return externDists


########################################################################################################################

if __name__ == '__main__':
  print(cdist([[1, 1, 1], [3, 3, 3], [5, 5, 5]], np.atleast_2d([2, 2, 2]), 'sqeuclidean').flatten())

  from scipy.stats import spearmanr

  print(spearmanr([1, 2, 3], [2, 4, 1]))
