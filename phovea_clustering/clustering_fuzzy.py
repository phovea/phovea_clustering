__author__ = 'Michael Kern'
__version__ = '0.0.3'
__email__ = 'kernm@in.tum.de'

########################################################################################################################
# libraries

# module to load own configurations
import phovea_server.config

# request config if needed for the future
config = phovea_server.config.view('caleydo-clustering')

# library to conduct matrix/vector calculus
import numpy as np

from clustering_util import similarityMeasurement


########################################################################################################################
# class definition

class Fuzzy(object):
  """
  Formulas: https://en.wikipedia.org/wiki/Fuzzy_clustering
  """

  def __init__(self, obs, numClusters, m=2.0, threshold=-1, distance='euclidean', init=None, error=0.0001):
    """
    Initializes algorithm.
    :param obs: observation matrix / genomic data
    :param numClusters: number of clusters
    :param m: fuzzifier, controls degree of fuzziness, from [1; inf]
    :return:
    """
    # observation
    self.__obs = np.nan_to_num(obs)

    self.__n = obs.shape[0]

    # fuzzifier value
    self.__m = np.float(m)
    # number of clusters
    self.__c = numClusters

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
      self.__threshold = 1.0 / numClusters
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

  def computeCentroid(self):
    """
    Compute the new centroids using the computed partition matrix.
    :return:
    """
    uM = self.__u ** self.__m

    sumDataWeights = np.dot(uM, self.__obs)
    if self.__obs.ndim == 1:
      m = 1
    else:
      m = self.__obs.shape[1]

    sumWeights = np.sum(uM, axis=1)
    # tile array (sum of weights repeated in every row)
    sumWeights = np.ones((m, 1)).dot(np.atleast_2d(sumWeights)).T

    if self.__obs.ndim == 1:
      sumWeights = sumWeights.flatten()

    # divide by total sum to get new centroids
    self.__centroids = sumDataWeights / sumWeights

  # ------------------------------------------------------------------------------------------------------------------

  def computeCoefficients(self):
    """
    Compute new partition matrix / weights describing the degree of membership of each patient to all clusters.
    :return:
    """

    # TODO you can also use cdist of scipy.spatial.distance module
    distMat = np.zeros((self.__c, self.__n))

    for ii in range(self.__c):
      distMat[ii] = similarityMeasurement(self.__obs, self.__centroids[ii], self.__distance)

    # set zero values to smallest values to prevent inf results
    distMat = np.fmax(distMat, np.finfo(np.float64).eps)

    # apply coefficient formula
    denom = np.float(self.__m - 1.0)
    self.__u = distMat ** (-2.0 / denom)

    sumCoeffs = np.sum(self.__u, axis=0)

    self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(sumCoeffs))
    self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)

  # ------------------------------------------------------------------------------------------------------------------

  def run(self):
    """
    Perform the c-means fuzzy clustering.
    :return:
    """
    MAX_ITER = 100
    iter = 0

    while iter < MAX_ITER:
      # save last partition matrix
      uOld = np.copy(self.__u)
      # compute centroids with given weights
      self.computeCentroid()
      # compute new coefficient matrix
      self.computeCoefficients()

      # normalize weight / partition matrix u
      self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(np.sum(self.__u, axis=0)))
      self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)

      # compute the difference between the old and new matrix
      epsilon = np.linalg.norm(self.__u - uOld)

      # stop if difference (epsilon) is smaller than the user-defined threshold
      if epsilon < self.__error:
        break

      iter += 1

    self.__end()

    u = self.__u.T
    # print(self.__u.T)

    return self.__centroids.tolist(), self.__clusterLabels, u.tolist(), self.__threshold

  # ------------------------------------------------------------------------------------------------------------------

  def __end(self):
    """
    Conduct the cluster assignments and creates clusterLabel array.
    :return:
    """
    # assign patient to clusters
    # transpose to get a (n, c) matrix
    u = self.__u.T

    self.__labels = np.zeros(self.__n, dtype=np.int)
    self.__clusterLabels = [[] for _ in range(self.__c)]
    # gather all probabilities / degree of memberships of each patient to the clusters
    # self.__clusterProbs = [[] for _ in range(self.__c)]
    # probability that the patients belongs to each cluster
    maxProb = np.float64(self.__threshold)

    for ii in range(self.__n):
      # clusterID = np.argmax(u[ii])
      # self.__labels = clusterID
      # self.__clusterLabels[clusterID].append(ii)

      for jj in range(self.__c):
        if u[ii][jj] >= maxProb:
          clusterID = jj
          self.__labels = clusterID
          self.__clusterLabels[clusterID].append(int(ii))

          # for ii in range(self.__c):
          #     self.__clusterLabels[ii], _ = computeClusterInternDistances(self.__obs, self.__clusterLabels[ii])


########################################################################################################################

def _plugin_initialize():
  """
  optional initialization method of this module, will be called once
  :return:
  """
  pass


# ----------------------------------------------------------------------------------------------------------------------

def create(data, numCluster, m, threshold, distance):
  """
  by convention contain a factory called create returning the extension implementation
  :return:
  """
  return Fuzzy(data, numCluster, m, threshold, distance)


########################################################################################################################

if __name__ == '__main__':
  data = np.array([[1, 1, 2], [5, 4, 5], [3, 2, 2], [8, 8, 7], [9, 8, 9], [2, 2, 2]])
  # data = np.array([1,1.1,5,8,5.2,8.3])

  fuz = Fuzzy(data, 3, 1.5)
  print(fuz.run())
