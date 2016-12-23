########################################################################################################################
# libraries

# module to load own configurations
import phovea_server.config


# library to conduct matrix/vector calculus
import numpy as np
# fastest distance computation by scipy
# import scipy.spatial as spt

# utility functions for clustering and creating the dendrogram trees
from clustering_util import BinaryNode, BinaryTree
from clustering_util import similarity_measurement_matrix
from clustering_util import compute_cluster_intern_distances
from clustering_util import cut_json_tree_by_clusters

__author__ = 'Michael Kern'
__version__ = '0.0.3'
__email__ = 'kernm@in.tum.de'
# request config if needed for the future
config = phovea_server.config.view('caleydo-clustering')


########################################################################################################################

class Hierarchical(object):
  """
  This is a implementation of hierarchical clustering on genomic data using the Lance-Williams dissimilarity update
  to compute different distance metrics (single linkage, complete linkage, ...).
  Lance-Williams explained in: http://arxiv.org/pdf/1105.0121.pdf
  """

  def __init__(self, obs, method='single', distance='euclidean'):
    """
    Initializes the algorithm
    :param obs: genomic data / matrix
    :param method: linkage method
    :return:
    """
    # genomic data / matrix
    # observations, can be 1D array or 2D matrix with genes as rows and conditions as columns
    # remove all NaNs in data
    self.__obs = np.nan_to_num(obs)

    num_genes = np.shape(self.__obs)[0]
    self.__n = num_genes

    # check if dimension is 2D
    # if self.__obs.ndim == 2:
    #     # obtain number of observations (rows)
    #     num_genes, _ = np.shape(self.__obs)
    #     self.__n = num_genes

    # else:
    #     print("[Error]:\tdata / observations must be 2D. 1D observation arrays are not supported")
    #     raise Attribute_error

    # distance measurement
    self.__distance = distance

    # distance / proximity matrix
    self.__d = []
    self.__compute_proximity_matrix()
    # dictionary mapping the string id (i,j,k,...) of clusters to corresponding index in matrix
    self.__id_map = {}
    # inverse mapping of id_map --> returns the string id given a certain index
    self.__key_map = {}
    # contains actual index of all clusters, old clusters are from [0, n - 1], new clusters have indices in range
    # [n, 2n - 1]
    self.__cluster_map = {}
    for ii in range(self.__n):
      self.__id_map[str(ii)] = ii
      self.__key_map[ii] = str(ii)
      self.__cluster_map[str(ii)] = ii

    # linkage method for hierarchical clustering
    self.__method = method

    # internal dendrogram tree
    self.__tree = None

  # ------------------------------------------------------------------------------------------------------------------

  def __call__(self):
    """
    Caller function for server API
    :return:
    """
    return self.run()

  # ------------------------------------------------------------------------------------------------------------------

  @property
  def tree(self):
    return self.__tree

  # ------------------------------------------------------------------------------------------------------------------

  def __get_coefficients(self, cluster_i, cluster_j):
    """
    Compute the coefficients for the Lance-Williams algorithm
    :param cluster_i:
    :param cluster_j:
    :return:
    """
    # TODO! use hash map for storing numbers instead of computing them every time
    if self.__method == 'single':
      return 0.5, 0.5, 0, -0.5
    elif self.__method == 'complete':
      return 0.5, 0.5, 0, 0.5
    elif self.__method == 'weighted':
      return 0.5, 0.5, 0, 0
    elif self.__method == 'median':
      return 0.5, 0.5, -0.25, 0

    # TODO! ATTENTION! average method should compute the cluster centroids using the average
    # TODO! || cluster_i - cluster_j || ** 2
    elif self.__method == 'average':
      n_i = np.float(cluster_i.count(',') + 1)
      n_j = np.float(cluster_j.count(',') + 1)
      sum_n = n_i + n_j
      return (n_i / sum_n), (n_j / sum_n), 0, 0

    # TODO! ATTENTION! centroid method should compute the cluster centroids using the mean
    # TODO! || cluster_i - cluster_j || ** 2
    elif self.__method == 'centroid':
      n_i = np.float(cluster_i.count(',') + 1)
      n_j = np.float(cluster_j.count(',') + 1)
      sum_n = n_i + n_j
      return (n_i / sum_n), (n_j / sum_n), -((n_i * n_j) / (sum_n ** 2)), 0

    # TODO! Support ward method
    # TODO! (|cluster_i| * |cluster_j|) / (|cluster_i| + |cluster_j) * || cluster_i - cluster_j || ** 2
    # elif self.__method == 'ward':
    #     n_i = np.float(cluster_i.count(',') + 1)
    #     n_j = np.float(cluster_j.count(',') + 1)
    #     n_k = np.float(cluster_k.count(',') + 1)
    #     sum_n = n_i + n_j + n_k
    #     return (n_i + n_k) / sum_n, (n_j + n_k) / sum_n, -n_k / sum_n, 0
    else:
      raise AttributeError

  # ------------------------------------------------------------------------------------------------------------------

  def __compute_proximity_matrix(self):
    """
    Compute the proximity of each observation and store the results in a nxn matrix
    :return:
    """

    # create distance matrix of size n x n
    self.__d = np.zeros((self.__n, self.__n))

    # compute euclidean distance
    # TODO! implement generic distance functions
    # TODO! look for an alternative proximity analysis without computing all distances
    self.__d = similarity_measurement_matrix(self.__obs, self.__distance)

    # get number of maximum value of float
    self.__max_value = self.__d.max() + 1

    # fill diagonals with max value to exclude them from min dist process
    # TODO! operate only on upper triangle matrix of distance matrix
    np.fill_diagonal(self.__d, self.__max_value)

    # print('\t-> finished.')

  # ------------------------------------------------------------------------------------------------------------------

  def __get_matrix_minimum_indices(self):
    """
    Searches for the minimum distance in the distance matrix
    :return: indices of both clusters having the smallest distance
    """
    min_dist = self.__d.min()
    min_list = np.argwhere(self.__d == min_dist)

    min_i, min_j = 0, 0

    # look for indices, where i < j
    # TODO! for the future --> use upper triangle matrix
    for ii in range(len(min_list)):
      min_i, min_j = min_list[ii]
      if min_i < min_j:
        break

    if min_i == min_j:
      print("ERROR")

    return self.__key_map[min_i], self.__key_map[min_j], min_dist

  # ------------------------------------------------------------------------------------------------------------------

  def __delete_clusters(self, i, j):
    """
    Reorders and reduces the matrix to insert the new cluster formed of cluster i and j
    and its distance values, and removes the old clusters by cutting the last row.
    :param i: cluster index i
    :param j: cluster index j
    :return:
    """
    id_i = self.__id_map[str(i)]
    id_j = self.__id_map[str(j)]

    # min_id = min(id_i, id_j)
    max_id = max(id_i, id_j)

    # now set column max ID to last column -> swap last and i column
    last_row = self.__d[self.__n - 1]
    self.__d[max_id] = last_row
    self.__d[:, max_id] = self.__d[:, (self.__n - 1)]

    # set key of last column (cluster) to column of the cluster with index max_id
    key = self.__key_map[self.__n - 1]
    self.__id_map[key] = max_id
    self.__key_map[max_id] = key

    # delete entries in id and key map --> not required anymore
    try:
      del self.__id_map[i]
      del self.__id_map[j]
      del self.__key_map[self.__n - 1]
    except KeyError:
      print("\nERROR: Key {} not found in id_map".format(j))
      print("ERROR: Previous key: {} in id_map".format(i))
      print("Given keys: ")
      for key in self.__id_map:
        print(key)
      return

    # reduce dimension of matrix by one column and row
    self.__n -= 1
    self.__d = self.__d[:-1, :-1]

  # ------------------------------------------------------------------------------------------------------------------

  def __merge_clusters(self, i, j):
    """
    Merges cluster i and j, computes the new ID and distances of the newly formed cluster
    and stores required information
    :param i: cluster index i
    :param j: cluster index j
    :return:
    """
    id_i = self.__id_map[str(i)]
    id_j = self.__id_map[str(j)]

    min_id = min(id_i, id_j)
    max_id = max(id_i, id_j)

    # use Lance-Williams formula to compute linkages
    dki = self.__d[:, min_id]
    dkj = self.__d[:, max_id]
    dij = self.__d[min_id, max_id]
    dist_ij = np.abs(dki - dkj)

    # compute coefficients
    ai, aj, b, y = self.__get_coefficients(i, j)

    new_entries = ai * dki + aj * dkj + b * dij + y * dist_ij
    new_entries[min_id] = self.__max_value
    new_entries[max_id] = self.__max_value

    # add new column and row
    self.__d[min_id] = new_entries
    self.__d[:, min_id] = new_entries

    id_ij = min_id
    new_key = i + ',' + j
    self.__id_map[new_key] = id_ij
    self.__key_map[id_ij] = new_key
    self.__cluster_map[new_key] = len(self.__cluster_map)

    # delete old clusters
    self.__delete_clusters(i, j)

    # count number of elements
    return new_key.count(',') + 1

  # ------------------------------------------------------------------------------------------------------------------

  def run(self):
    """
    Conducts the algorithm until there's only one cluster.
    :return:
    """

    # number of the current iteration
    m = 0

    # resulting matrix containing information Z[i,x], x=0: cluster i, x=1: cluster j, x=2: dist(i,j), x=3: num(i,j)
    runs = self.__n - 1
    z = np.array([[0 for _ in range(4)] for _ in range(runs)], dtype=np.float)

    while m < runs:
      m += 1

      i, j, dist_ij = self.__get_matrix_minimum_indices()
      num_ij = self.__merge_clusters(i, j)

      cluster_i, cluster_j = self.__cluster_map[i], self.__cluster_map[j]
      z[m - 1] = [int(min(cluster_i, cluster_j)), int(max(cluster_i, cluster_j)), np.float(dist_ij), int(num_ij)]

    # reset number n to length of first dimension (number of genes)
    self.__n = np.shape(self.__obs)[0]

    self.__tree = self.generate_tree(z)
    return z.tolist()

  # ------------------------------------------------------------------------------------------------------------------

  def generate_tree(self, linkage_matrix):
    """
    Computes the dendrogram tree for a given linkage matrix.
    :param linkage_matrix:
    :return:
    """
    self.__tree = None

    tree_map = {}
    num_trees = len(linkage_matrix)

    for ii in range(num_trees):
      entry = linkage_matrix[ii]
      current_id = self.__n + ii
      left_index, right_index, value = int(entry[1]), int(entry[0]), entry[2], int(entry[3])
      left = right = None

      if left_index < self.__n:
        left = BinaryNode(self.__obs[left_index].tolist(), left_index, 1, None, None)
      else:
        left = tree_map[left_index]

      if right_index < self.__n:
        right = BinaryNode(self.__obs[right_index].tolist(), right_index, 1, None, None)
      else:
        right = tree_map[right_index]

      if isinstance(left, BinaryNode) and isinstance(right, BinaryNode):
        tree_map[current_id] = BinaryTree(left, right, current_id, value)
      elif isinstance(left, BinaryNode):
        tree_map[current_id] = right.add_node(left, current_id, value)
        del tree_map[right_index]
      elif isinstance(right, BinaryNode):
        tree_map[current_id] = left.add_node(right, current_id, value)
        del tree_map[left_index]
      else:
        tree_map[current_id] = left.merge(right, current_id, value)
        del tree_map[right_index]
        del tree_map[left_index]

    self.__tree = tree_map[num_trees + self.__n - 1]
    return self.__tree

    # ------------------------------------------------------------------------------------------------------------------


########################################################################################################################


def get_clusters(k, obs, dendrogram, sorted=True):
  """
  First implementation to cut dendrogram tree automatically by choosing nodes having the greatest node values
  or rather distance to the other node / potential cluster
  :param k: number of desired clusters
  :param obs: set of observations
  :param dendrogram: dendrogram tree
  :return: centroids, sorted cluster labels and normal label list
  """
  obs = np.nan_to_num(obs)
  n = obs.shape[0]

  if isinstance(dendrogram, BinaryTree):
    cluster_labels = dendrogram.cut_tree_by_clusters(k)
  else:
    cluster_labels = cut_json_tree_by_clusters(dendrogram, k)

  cluster_centroids = []
  labels = np.zeros(n, dtype=np.int)
  cluster_id = 0

  for ii in range(len(cluster_labels)):
    cluster = cluster_labels[ii]
    sub_obs = obs[cluster]
    cluster_centroids.append(np.mean(sub_obs, axis=0).tolist())

    for id in cluster:
      labels[id] = cluster_id

    # sort labels according to their distance
    if sorted:
      cluster_labels[ii], _ = compute_cluster_intern_distances(obs, cluster)

    cluster_id += 1

  return cluster_centroids, cluster_labels, labels.tolist()


########################################################################################################################

def _plugin_initialize():
  """
  optional initialization method of this module, will be called once
  :return:
  """
  pass


# ----------------------------------------------------------------------------------------------------------------------

def create(data, method, distance):
  """
  by convention contain a factory called create returning the extension implementation
  :return:
  """
  return Hierarchical(data, method, distance)


########################################################################################################################
def _main():
  from timeit import default_timer as timer
  # from scipy.cluster.hierarchy import linkage, leaves_list

  np.random.seed(200)
  # data = np.array([[1,2,3],[5,4,5],[3,2,2],[8,8,7],[9,6,7],[2,3,4]])
  data = np.array([1, 1.1, 5, 8, 5.2, 8.3])

  time_mine = 0
  time_theirs = 0

  n = 10

  for i in range(n):
    # data = np.array([np.random.rand(6000) * 4 - 2 for _ in range(249)])
    # import time
    s1 = timer()
    hier = Hierarchical(data, 'complete')
    # s = time.time()
    linkage_matrix = hier.run()
    e1 = timer()
    print(linkage_matrix)
    tree = hier.generate_tree(linkage_matrix)
    # print(tree.get_leaves())
    # print(tree.jsonify())
    # print(hier.get_clusters(3))
    import json

    json_tree = json.loads(tree.jsonify())
    get_clusters(3, data, json_tree)

    s2 = timer()
    # linkage_matrix2 = linkage(data, 'complete')
    # print(leaves_list(linkage_matrix2))
    e2 = timer()

    time_mine += e1 - s1
    time_theirs += e2 - s2

  # print(linkage_matrix)
  # print(linkage_matrix2)
  print('mine: {}'.format(time_mine / n))
  print('theirs: {}'.format(time_theirs / n))

if __name__ == '__main__':
  _main()
