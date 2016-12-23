########################################################################################################################

import random
import numpy as np

# use scipy to compute different distance matrices
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats

__author__ = "Michael Kern"
__email__ = 'kernm@in.tum.de'

"""
http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
--> good explanation to create weighted choices / random numbers
"""


def weighted_choice(weights):
  # compute sum of all weights
  sum_total = sum(weights)
  # compute a random with range[0, sum_total]
  rnd = random.random() * sum_total

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
  def __init__(self, value, id, size, left_child, right_child):
    self.value = value
    self.left = left_child
    self.right = right_child
    self.size = size
    self.id = id
    self.parent = None
    self.indices = [id]

    # create json info on the fly
    self.json = {"id": self.id, "size": self.size, "value": self.value, "indices": [id]}
    if left_child is not None and right_child is not None:
      # self.json["value"] = np.mean(self.value)
      self.json["children"] = [right_child.json, left_child.json]
      self.indices = [] + right_child.indices + left_child.indices
      self.json["indices"] = self.indices

  def is_leave(self):
    return self.left is None and self.right is None


########################################################################################################################

"""
Implementation of an hierarchical binary tree
"""


class BinaryTree:
  # this tree must not be empty and must have at least two children (leaves)
  def __init__(self, left_node, right_node, new_id, new_value):
    self.__create_new_root(left_node, right_node, new_id, new_value)

  # ------------------------------------------------------------------------------------------------------------------

  def add_node(self, new_node, new_id, new_value):
    self.__create_new_root(self.root, new_node, new_id, new_value)
    return self

  # ------------------------------------------------------------------------------------------------------------------

  def merge(self, tree, new_id, new_value):
    self.__create_new_root(self.root, tree.root, new_id, new_value)
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

  def cut_tree_by_clusters(self, k):
    queue = [self.root]

    while len(queue) < k:
      node = queue.pop(0)
      queue.append(node.left)
      queue.append(node.right)

      def key_func(x):
        if x.is_leave():
          return 0
        else:
          return -x.value

      queue.sort(key=key_func)

    clusters = []

    for node in queue:
      clusters.append(node.indices)

    return clusters

  # ------------------------------------------------------------------------------------------------------------------

  def __traverse_json(self, node):
    json = {"id": node.id, "size": node.size, "value": node.value}
    if node.left is None and node.right is None:
      return json
    else:
      json["children"] = [] + [self.__traverse_json(node.left)] + [self.__traverse_json(node.right)]

    return json

  # ------------------------------------------------------------------------------------------------------------------

  def get_leaves(self):
    return self.root.indices
    # return self.__traverseIDs(self.root)

  # ------------------------------------------------------------------------------------------------------------------

  def __traverse_ids(self, node):

    if node.left is None and node.right is None:
      return [node.id]
    else:
      return [] + self.__traverse_ids(node.right) + self.__traverse_ids(node.left)

  # ------------------------------------------------------------------------------------------------------------------

  def __create_new_root(self, left_node, right_node, new_id, new_value):
    new_size = left_node.size + right_node.size
    self.root = BinaryNode(new_value, new_id, new_size, left_node, right_node)
    left_node.parent = right_node.parent = self.root

    # ------------------------------------------------------------------------------------------------------------------


def cut_json_tree_by_clusters(json_data, k):
  # import json
  # tree = json.loads(json_data)
  queue = [json_data]

  while len(queue) < k:
    node = queue.pop(0)
    queue.append(node['children'][0])
    queue.append(node['children'][1])

    def key_func(x):
      if 'children' not in x:
        return 0
      else:
        return -x['value']

    queue.sort(key=key_func)

  clusters = []

  for node in queue:
    clusters.append(node['indices'])

  return clusters


########################################################################################################################

def euclidean_distance(matrix, vector, squared=False):
  """
  Computes the euclidean distance between a vector and the rows of a matrix in parallel.
  :param matrix: array of observations or clusters
  :param vector: cluster centroid or observation
  :return:
  """

  # compute distance between values in matrix and the vector
  dist_mat = matrix - vector
  num_values = len(matrix)
  distances = np.array([0.0 for _ in range(num_values)], dtype=np.float)

  for ii in range(num_values):
    distance = dist_mat[ii]
    # always try to use np.dot when computing euclidean distance
    # it's way faster than ** 2 and sum(..., axis=1)
    distances[ii] = np.dot(distance, distance)

  if squared:
    return distances
  else:
    return np.sqrt(distances)


# ----------------------------------------------------------------------------------------------------------------------

def correlation_distance(matrix, vector, method):
  """

  :param matrix:
  :param vector:
  :return:
  """

  num_values = len(matrix)
  distances = np.array([0.0 for _ in range(num_values)], dtype=np.float)

  for ii in range(num_values):
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


def similarity_measurement(matrix, vector, method='euclidean'):
  from scipy.spatial.distance import cdist

  if method == 'euclidean':
    return euclidean_distance(matrix, vector)

  if method == 'sqeuclidean':
    return euclidean_distance(matrix, vector, True)

  spatial_methods = ['cityblock', 'chebyshev', 'canberra', 'correlation', 'hamming', 'mahalanobis', ]

  if method in spatial_methods:
    return np.nan_to_num(cdist(matrix, np.atleast_2d(vector), method).flatten())

  corr_methods = ['spearman', 'pearson', 'kendall']

  if method in corr_methods:
    return correlation_distance(matrix, vector, method)

  raise AttributeError


# ----------------------------------------------------------------------------------------------------------------------

def euclidean_distance_matrix(matrix, squared=False):
  """
  Compute the euclidean distance matrix required for the algorithm
  :param matrix:
  :param n:
  :return:
  """

  n = np.shape(matrix)[0]
  dist_mat = np.zeros((n, n))

  # use Gram matrix and compute distances without inner products | FASTER than row-by-row method
  "Gramiam matrix to compute dot products of each pair of elements: "
  "<https://en.wikipedia.org/wiki/Gramian_matrix>"
  gram_mat = np.zeros((n, n))
  for ii in range(n):
    for jj in range(ii, n):
      gram_mat[ii, jj] = np.dot(matrix[ii], matrix[jj])

  # # ! This is slower than computing dot products of rows manually in python
  # # ! And we only require the upper triangle matrix of the Gram matrix
  # gram_mat = np.dot(self.__obs, self.__obs.T)

  # make use of formula |a - b|^2 = a^2 - 2ab + b^2
  for ii in range(n):
    # self.__d[ii, ii] = self.__maxValue
    jj = np.arange(ii + 1, n)
    dist_mat[ii, jj] = gram_mat[ii, ii] - 2 * gram_mat[ii, jj] + gram_mat[jj, jj]
    dist_mat[jj, ii] = dist_mat[ii, jj]

  # # take square root of distances to compute real euclidean distance
  # dist_mat = np.sqrt(dist_mat)

  "alternative version --> use scipy's fast euclidean distance implementation: FASTEST"
  # dist_mat = spt.distance.pdist(self.__obs, 'euclidean')
  # self.__d = spt.distance.squareform(dist_mat)
  # print(dist_mat)

  if squared:
    return dist_mat
  else:
    return np.sqrt(dist_mat)


# ----------------------------------------------------------------------------------------------------------------------

def norm1_distance(matrix, vector):
  """
  Computes the norm-1 distance between a vector and the rows of a matrix in parallel.
  :param matrix: array of observations or clusters
  :param vector: cluster centroid or observation
  :return:
  """
  dist_mat = np.abs(matrix - vector)
  num_values = len(vector)

  distances = np.sum(dist_mat, axis=1) / num_values
  return distances


# ----------------------------------------------------------------------------------------------------------------------

def pearson_correlation_matrix(matrix):
  """

  :param matrix:
  :param n:
  :return:
  """
  # TODO! other possibilites like 1 - abs(corr) | sqrt(1 - corr ** 2) | (1 - corr) / 2
  dist_mat = 1 - np.corrcoef(matrix)

  return dist_mat


# ----------------------------------------------------------------------------------------------------------------------

def stats_correlation_matrix(matrix, method):
  if method == 'pearson':
    return pearson_correlation_matrix(matrix)

  n = np.shape(matrix)[0]
  dist_mat = np.zeros((n, n))

  for ii in range(n):
    row_i = matrix[ii]
    for jj in range(ii + 1, n):
      row_j = matrix[jj]
      corr = 0

      if method == 'spearman':
        corr, _ = stats.spearmanr(row_i, row_j)

      if method == 'kendall':
        corr, _ = stats.kendalltau(row_i, row_j)

      # TODO! other possibilites like 1 - abs(corr) | sqrt(1 - corr ** 2) | (1 - corr) / 2
      corr = 1 - corr

      dist_mat[ii, jj] = corr
      dist_mat[jj, ii] = corr

  return dist_mat


# ----------------------------------------------------------------------------------------------------------------------

def similarity_measurement_matrix(matrix, method):
  """
  Generic function to determine the similarity measurement for clustering
  :param matrix:
  :param method:
  :return:
  """
  if method == 'euclidean':
    return euclidean_distance_matrix(matrix)
    # return squareform(pdist(matrix, method))

  if method == 'sqeuclidean':
    return euclidean_distance_matrix(matrix, True)
    # return squareform(pdist(matrix, method))

  spatial_methods = ['cityblock', 'chebyshev', 'canberra', 'correlation', 'hamming', 'mahalanobis']

  if method in spatial_methods:
    return squareform(np.nan_to_num(pdist(matrix, method)))

  corr_methods = ['spearman', 'pearson', 'kendall']

  if method in corr_methods:
    return stats_correlation_matrix(matrix, method)

  raise AttributeError


########################################################################################################################
# utility functions to compute distances between rows and cluster centroids

def compute_cluster_intern_distances(matrix, labels, sorted=True, metric='euclidean'):
  """
  Computes the distances of each element in one cluster to the cluster's centroid. Returns distance values and labels
  sorted in ascending order.
  :param matrix:
  :param labels:
  :return: labels / indices of elements corresponding to distance array, distance values of cluster
  """
  cluster_labels = np.array(labels)
  if len(cluster_labels) == 0:
    return [], []

  sub_matrix = matrix[cluster_labels]
  # compute centroid of cluster along column (as we want to average each gene separately)
  centroid = np.mean(sub_matrix, axis=0)

  # compute distances to centroid
  dists = similarity_measurement(sub_matrix, centroid, metric)

  if sorted == 'true':
    # sort values
    indices = range(len(dists))
    indices.sort(key=dists.__getitem__)
    dists.sort()

    # reverse order if correlation coefficient is used
    # (1 means perfect correlation while -1 denotes opposite correlation)
    corr_metrics = ['pearson', 'spearman', 'kendall']
    if metric in corr_metrics:
      indices.reverse()
      dists = dists[::-1]

    # write back to our arrays
    dist_labels = cluster_labels[indices].tolist()
    dist_values = dists.tolist()
  else:
    dist_labels = cluster_labels.tolist()
    dist_values = dists.tolist()

  return dist_labels, dist_values


# ----------------------------------------------------------------------------------------------------------------------

def compute_cluster_extern_distances(matrix, labels, outer_labels, metric='euclidean'):
  """
  Compute the distances of patients in one cluster to the centroids of all other clusters.
  :param matrix:
  :param labels:
  :param outer_labels:
  :return:
  """
  extern_dists = []
  intern_sub_matrix = matrix[labels]

  for extern_labels in outer_labels:

    if len(extern_labels) == 0:
      extern_dists.append([])

    # compute centroid of external cluster
    sub_matrix = matrix[extern_labels]
    centroid = np.mean(sub_matrix, axis=0)

    dists = similarity_measurement(intern_sub_matrix, centroid, metric)
    extern_dists.append(dists.tolist())

  return extern_dists


########################################################################################################################

if __name__ == '__main__':
  from scipy.spatial.distance import cdist
  print(cdist([[1, 1, 1], [3, 3, 3], [5, 5, 5]], np.atleast_2d([2, 2, 2]), 'sqeuclidean').flatten())

  from scipy.stats import spearmanr

  print(spearmanr([1, 2, 3], [2, 4, 1]))
