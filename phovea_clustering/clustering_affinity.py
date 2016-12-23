########################################################################################################################
# libraries

# module to load own configurations
import phovea_server.config
import numpy as np
from clustering_util import similarity_measurementmatrix
from timeit import default_timer as timer

# request config if needed for the future
config = phovea_server.config.view('caleydo-clustering')

__author__ = 'Michael Kern'
__version__ = '0.0.1'
__email__ = 'kernm@in.tum.de'


########################################################################################################################

class AffinityPropagation:
  """
  This is an implementation of the affinity propagation algorithm to cluster genomic data / matrices.
  Implementation details: <http://www.psi.toronto.edu/index.php?q=affinity%20propagation>.
  Matlab implementation: <http://www.psi.toronto.edu/affinitypropagation/software/apcluster.m>
  Returns the centroids and labels / stratification of each row belonging to one cluster.
  """

  def __init__(self, obs, damping=0.5, factor=1.0, pref_method='minimum', distance='euclidean'):
    """
    Initializes the algorithm.
    :param obs: genomic data / matrix
    :param damping: controls update process to dampen oscillations
    :param factor: controls the preference value (influences number of clusters)
    :param pref_method: all points are chosen equally with a given preference (median or minimum of similarity matrix)
    :return:
    """
    self.__n = np.shape(obs)[0]
    # observations, can be 1D array or 2D matrix with genes as rows and conditions as columns
    # remove all NaNs in data
    self.__obs = np.nan_to_num(obs)
    # variables influencing output of clustering algorithm
    self.__damping = damping
    self.__factor = factor
    self.__prev_method = pref_method

    # similarity matrix
    self.__S = np.zeros((self.__n, self.__n))
    # availability matrix
    self.__A = np.zeros((self.__n, self.__n))
    # responsibility matrix
    self.__R = np.zeros((self.__n, self.__n))

    self.min_value = np.finfo(np.float).min

    # self.__mx1 = np.full(self.__n, self.min_value)
    # self.__mx2 = np.full(self.__n, self.min_value)

    self.__idx = np.zeros(self.__n)

    # set similarity computation
    self.__distance = distance

    self.__compute_similarity()

  # ------------------------------------------------------------------------------------------------------------------

  def __call__(self):
    """
    Caller function for server API.
    """
    return self.run()

  # ------------------------------------------------------------------------------------------------------------------

  def __compute_similarity(self):
    """
    Compute the similarity matrix from the original observation matrix and set preference of each element.
    :return: _similarity matrix
    """
    # compute distance matrix containing the negative sq euclidean distances -|| xi - xj ||**2
    self.__S = -similarity_measurementmatrix(self.__obs, self.__distance)

    # determine the preferences S(k,k) to control the output of clusters
    pref = 0
    # could be median or minimum
    if self.__prev_method == 'median':
      pref = float(np.median(self.__S)) * self.__factor
    elif self.__prev_method == 'minimum':
      pref = np.min(self.__S) * self.__factor
    else:
      raise AttributeError

    np.fill_diagonal(self.__S, pref)

  # ------------------------------------------------------------------------------------------------------------------

  def run(self):
    """
    Runs the algorithm of affinity propagation. Conducts at least 100 iterations and checks if the outcome of
    current exemplars/clusters has converged. If not, the algorithm will continue until convergence is found
    or the maximum number of iterations (200) is reached.
    :return:
    """
    max_iter = 200
    max_conv_iter = 100

    # sum all decisions for exemplars per round
    decision_sum = np.zeros(self.__n)
    # collect decisions for one exemplar per iteration round
    decision_iter = np.zeros((max_conv_iter, self.__n))
    # counter for decisions (= consider data element as exemplar in each algorithm iteration)
    decision_counter = max_conv_iter
    # indicates if algorithm has converged
    is_converged = False

    centroids = []
    it = 0
    cluster_i = []

    # helpful variables (that do not need recomputation)
    index_diag = np.arange(self.__n)
    indices_diag = np.diag_indices_from(self.__R)
    new_a = np.zeros((self.__n, self.__n))
    new_r = np.zeros((self.__n, self.__n))

    for it in range(1, max_iter + 1):

      # ----------------------------------------------------------------------------------------------------------

      # compute responsibility matrix
      m_as = self.__A + self.__S

      max_y = np.max(m_as, axis=1)
      index_y = np.argmax(m_as, axis=1)

      # set values of maxima to zero in m_as matrix
      m_as[index_diag, index_y] = self.min_value

      # look for second maxima
      max_y2 = np.max(m_as, axis=1)

      # perform responsibility update
      for ii in range(self.__n):
        # s(i, k) - max({ a(i, k') + s(i, k') })
        new_r[ii] = self.__S[ii] - max_y[ii]

      # subtract second maximum from row -> column entry with maximum value
      new_r[index_diag, index_y] = self.__S[index_diag, index_y] - max_y2[index_diag]

      # dampen values
      # self.__R = self.__damping * self.__R + (1 - self.__damping) * new_r
      self.__R *= self.__damping
      self.__R += (1 - self.__damping) * new_r

      # ----------------------------------------------------------------------------------------------------------

      # compute availability matrix
      # cut out negative elements
      # TODO! slow because of copy operation
      rp = np.maximum(self.__R, 0)

      # write back all diagonal elements als self representatives
      rp[indices_diag] = self.__R[indices_diag]
      sum_cols = np.sum(rp, axis=0)

      # apply availability update
      new_a[:, ] = sum_cols
      new_a -= rp
      # for ii in range(self.__n):
      #     # r(k, k) + sum(max(0, r(i',k))
      #     new_a[:, ii] = sum_cols[ii] - Rp[:, ii]

      diag_a = np.diag(new_a)
      # take minimum of all the values in A, cut out all values above zero
      # new_a = np.minimum(new_a, 0)
      new_a[new_a > 0] = 0
      new_a[indices_diag] = diag_a[index_diag]

      # dampen values
      # self.__A = self.__damping * self.__A + (1 - self.__damping) * new_a
      self.__A *= self.__damping
      self.__A += (1 - self.__damping) * new_a

      # ----------------------------------------------------------------------------------------------------------

      # find exemplars for new clusters
      # old version which is slower
      # E = self.__R + self.__A
      # diag_e = np.diag(E)

      # take the diagonal elements of the create matrix E
      diag_e = np.diag(self.__R) + np.diag(self.__A)

      # all elements > 0 are considered to be an appropriate exemplar for the dataset
      cluster_i = np.argwhere(diag_e > 0).flatten()

      # count the number of clusters
      num_clusters = len(cluster_i)

      # ----------------------------------------------------------------------------------------------------------

      decision_counter += 1
      if decision_counter >= max_conv_iter:
        decision_counter = 0

      # subtract outcome of previous iteration (< 100) from the total sum of the decisions
      decision_sum -= decision_iter[decision_counter]

      decision_iter[decision_counter].fill(0)
      decision_iter[decision_counter][cluster_i] = 1

      # compute sum of decisions for each element being a exemplar
      decision_sum += decision_iter[decision_counter]

      # check for convergence
      if it >= max_conv_iter or it >= max_iter:
        is_converged = True

        for ii in range(self.__n):
          # if element is considered to be an exemplar in at least one iterations
          # and total of decisions in the last 100 iterations is not 100 --> no convergence
          if decision_sum[ii] != 0 and decision_sum[ii] != max_conv_iter:
            is_converged = False
            break

        if is_converged and num_clusters > 0:
          break

    # --------------------------------------------------------------------------------------------------------------

    # obtain centroids
    centroids = self.__obs[cluster_i]

    # find maximum columns in m_as matrix to assign elements to clusters / exemplars
    # fill A with negative values
    self.__A.fill(self.min_value)
    # set values of clusters to zero (as we only want to regard these values
    self.__A[:, cluster_i] = 0.0
    # fill diagonal of similarity matrix to zero (remove preferences)
    np.fill_diagonal(self.__S, 0.0)

    # compute m_as matrix
    m_as = self.__A + self.__S
    # since values are < 0, look for the maximum number in each row and return its column index
    self.__idx = np.argmax(m_as, axis=1)

    cluster_i = cluster_i.tolist()
    cluster_labels = [[] for _ in range(num_clusters)]

    # create labels per cluster
    for ii in range(self.__n):
      index = cluster_i.index(self.__idx[ii])
      self.__idx[ii] = index
      cluster_labels[index].append(ii)

    # return sorted cluster labels (that's why we call compute cluster distances, might be redundant)
    # for ii in range(num_clusters):
    #     cluster_labels[ii], _ = compute_cluster_intern_distances(self.__obs, cluster_labels[ii])

    # if is_converged:
    #     print('Algorithm has converged after {} iterations'.format(it))
    # else:
    #     print('Algorithm has not converged after 200 iterations')
    #
    # print('Number of detected clusters {}'.format(num_clusters))
    # print('Centroids: {}'.format(centroids))

    return centroids.tolist(), self.__idx.tolist(), cluster_labels


########################################################################################################################

def _plugin_initialize():
  """
  optional initialization method of this module, will be called once
  :return:
  """
  pass


# ----------------------------------------------------------------------------------------------------------------------

def create(data, damping, factor, preference, distance):
  """
  by convention contain a factory called create returning the extension implementation
  :return:
  """
  return AffinityPropagation(data, damping, factor, preference, distance)


########################################################################################################################

# from timeit import default_timer as timer

if __name__ == '__main__':
  np.random.seed(200)
  # data = np.array([[1,2,3],[5,4,5],[3,2,2],[8,8,7],[9,6,7],[2,3,4]])
  # data = np.array([np.random.rand(8000) * 4 - 2 for _ in range(500)])
  # data = np.array([[0.9],[1],[1.1],[10],[11],[12],[20],[21],[22]])
  data = np.array([1, 1.1, 5, 8, 5.2, 8.3])

  s = timer()
  aff = AffinityPropagation(data, 0.9, 1.0, 'median', 'euclidean')
  result = aff.run()
  e = timer()
  print(result)
  print('time elapsed: {}'.format(e - s))
