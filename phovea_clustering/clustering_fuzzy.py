########################################################################################################################
# libraries

# module to load own configurations
import phovea_server.config
# library to conduct matrix/vector calculus
import numpy as np
from clustering_util import similarity_measurement

# request config if needed for the future
config = phovea_server.config.view('caleydo-clustering')

__author__ = 'Michael Kern'
__version__ = '0.0.3'
__email__ = 'kernm@in.tum.de'


########################################################################################################################
# class definition

class Fuzzy(object):
  """
  Formulas: https://en.wikipedia.org/wiki/Fuzzy_clustering
  """

  def __init__(self, obs, num_clusters, m=2.0, threshold=-1, distance='euclidean', init=None, error=0.0001):
    """
    Initializes algorithm.
    :param obs: observation matrix / genomic data
    :param num_clusters: number of clusters
    :param m: fuzzifier, controls degree of fuzziness, from [1; inf]
    :return:
    """
    # observation
    self.__obs = np.nan_to_num(obs)

    self.__n = obs.shape[0]

    # fuzzifier value
    self.__m = np.float(m)
    # number of clusters
    self.__c = num_clusters

    # matrix u containing all the weights describing the degree of membership of each patient to the centroid
    if init is None:
      init = np.random.rand(self.__c, self.__n)

    self.__u = np.copy(init)

    # TODO! scikit normalizes the values at the beginning and at each step to [0; 1]
    self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(np.sum(self.__u, axis=0))).astype(np.float64)
    # remove all zero values and set them to smallest possible value
    self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)
    # centroids
    self.__centroids = np.zeros(self.__c)
    # threshold for stopping criterion
    self.__error = error
    # distance function
    self.__distance = distance

    # threshold or minimum probability used for cluster assignments
    if threshold == -1:
      self.__threshold = 1.0 / num_clusters
    else:
      self.__threshold = threshold

  # ------------------------------------------------------------------------------------------------------------------

  def __call__(self):
    """
    Caller function for server API
    :return:
    """
    return self.run()

  # ------------------------------------------------------------------------------------------------------------------

  def compute_centroid(self):
    """
    Compute the new centroids using the computed partition matrix.
    :return:
    """
    u_m = self.__u ** self.__m

    sum_data_weights = np.dot(u_m, self.__obs)
    if self.__obs.ndim == 1:
      m = 1
    else:
      m = self.__obs.shape[1]

    sum_weights = np.sum(u_m, axis=1)
    # tile array (sum of weights repeated in every row)
    sum_weights = np.ones((m, 1)).dot(np.atleast_2d(sum_weights)).T

    if self.__obs.ndim == 1:
      sum_weights = sum_weights.flatten()

    # divide by total sum to get new centroids
    self.__centroids = sum_data_weights / sum_weights

  # ------------------------------------------------------------------------------------------------------------------

  def compute_coefficients(self):
    """
    Compute new partition matrix / weights describing the degree of membership of each patient to all clusters.
    :return:
    """

    # TODO you can also use cdist of scipy.spatial.distance module
    dist_mat = np.zeros((self.__c, self.__n))

    for ii in range(self.__c):
      dist_mat[ii] = similarity_measurement(self.__obs, self.__centroids[ii], self.__distance)

    # set zero values to smallest values to prevent inf results
    dist_mat = np.fmax(dist_mat, np.finfo(np.float64).eps)

    # apply coefficient formula
    denom = np.float(self.__m - 1.0)
    self.__u = dist_mat ** (-2.0 / denom)

    sum_coeffs = np.sum(self.__u, axis=0)

    self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(sum_coeffs))
    self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)

  # ------------------------------------------------------------------------------------------------------------------

  def run(self):
    """
    Perform the c-means fuzzy clustering.
    :return:
    """
    max_iter = 100
    iter = 0

    while iter < max_iter:
      # save last partition matrix
      u_old = np.copy(self.__u)
      # compute centroids with given weights
      self.compute_centroid()
      # compute new coefficient matrix
      self.compute_coefficients()

      # normalize weight / partition matrix u
      self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(np.sum(self.__u, axis=0)))
      self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)

      # compute the difference between the old and new matrix
      epsilon = np.linalg.norm(self.__u - u_old)

      # stop if difference (epsilon) is smaller than the user-defined threshold
      if epsilon < self.__error:
        break

      iter += 1

    self.__end()

    u = self.__u.T
    # print(self.__u.T)

    return self.__centroids.tolist(), self.__cluster_labels, u.tolist(), self.__threshold

  # ------------------------------------------------------------------------------------------------------------------

  def __end(self):
    """
    Conduct the cluster assignments and creates cluster_label array.
    :return:
    """
    # assign patient to clusters
    # transpose to get a (n, c) matrix
    u = self.__u.T

    self.__labels = np.zeros(self.__n, dtype=np.int)
    self.__cluster_labels = [[] for _ in range(self.__c)]
    # gather all probabilities / degree of memberships of each patient to the clusters
    # self.__cluster_probs = [[] for _ in range(self.__c)]
    # probability that the patients belongs to each cluster
    max_prob = np.float64(self.__threshold)

    for ii in range(self.__n):
      # cluster_iD = np.argmax(u[ii])
      # self.__labels = cluster_iD
      # self.__cluster_labels[cluster_iD].append(ii)

      for jj in range(self.__c):
        if u[ii][jj] >= max_prob:
          cluster_id = jj
          self.__labels = cluster_id
          self.__cluster_labels[cluster_id].append(int(ii))

          # for ii in range(self.__c):
          #     self.__cluster_labels[ii], _ = compute_cluster_intern_distances(self.__obs, self.__cluster_labels[ii])


########################################################################################################################

def _plugin_initialize():
  """
  optional initialization method of this module, will be called once
  :return:
  """
  pass


# ----------------------------------------------------------------------------------------------------------------------

def create(data, num_cluster, m, threshold, distance):
  """
  by convention contain a factory called create returning the extension implementation
  :return:
  """
  return Fuzzy(data, num_cluster, m, threshold, distance)


########################################################################################################################

if __name__ == '__main__':
  data = np.array([[1, 1, 2], [5, 4, 5], [3, 2, 2], [8, 8, 7], [9, 8, 9], [2, 2, 2]])
  # data = np.array([1,1.1,5,8,5.2,8.3])

  fuz = Fuzzy(data, 3, 1.5)
  print(fuz.run())
