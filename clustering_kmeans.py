__author__ = 'Michael Kern'
__version__ = '0.0.2'
__email__ = 'kernm@in.tum.de'

########################################################################################################################
# libraries

# module to load own configurations
import caleydo_server.config
# request config if needed in the future
config = caleydo_server.config.view('caleydo-clustering')

# numpy important to conduct matrix/vector calculus
import numpy as np
# creates random numbers
import random

# contains utility functions
from clustering_util import weightedChoice, similarityMeasurement, computeClusterInternDistances

########################################################################################################################

class KMeans:
    """
    This is an implementation of the k-means algorithm to cluster genomic data / matrices.
    Returns the centroids, the labels / stratification of each row belonging to one cluster,
    distance matrix for cluster-cluster distance and distance arrays for row-clusterCentroid distance.
    Implementation detail: <https://en.wikipedia.org/wiki/K-means_clustering>
    """

    def __init__(self, obs, k, initMode='kmeans++', iters=1000, compare='sqeuclidean'):
        """
        Initializes the algorithm with observation, number of k clusters, the initial method and
        the maximum number of iterations.
        Initialization method of random cluster choice can be: forgy, uniform, random, plusplus
        :param obs: genomic data / matrix
        :param k: number of clusters
        :param initMode: initialization method
        :param iters: number of maximum iterations
        :return:
        """

        # number of clusters
        self.__k = k
        # observations, can be 1D array or 2D matrix with genes as rows and conditions as columns
        # remove all NaNs in data
        self.__obs = np.nan_to_num(obs)
        # number of observations / genes
        self.__n = np.shape(obs)[0]
        # maps the element ids to clusters
        self.__labelMap = np.zeros(self.__n, dtype=np.int)
        # cluster means and number of elements
        self.__clusterMeans = np.array([obs[0] for _ in range(k)], dtype=np.float)
        self.__clusterNums = np.array([0 for _ in range(k)], dtype=np.int)
        # tells if any cluster has changed or rather if any data item was moved
        self.__changed = True
        # number of iterations
        self.__iters = iters
        # initialization method
        self.__initMode = initMode
        # compare function
        self.__compare = compare

    # ------------------------------------------------------------------------------------------------------------------

    def __call__(self):
        """
        Caller function for server API.
        """
        return self.run()

    # ------------------------------------------------------------------------------------------------------------------

    def __init(self):
        """
        Initialize clustering with random clusters using a user-specified method
        :return:
        """
        # TODO! consider to init k-Means algorithm with Principal Component Analysis (PCA)
        # TODO! see <http://www.vision.caltech.edu/wikis/EE148/images/c/c2/KmeansPCA1.pdf>
        # init cluster
        if self.__initMode == 'forgy':
            self.__forgyMethod()
        elif self.__initMode == 'uniform':
            self.__uniformMethod()
        elif self.__initMode == 'random':
            self.__randomMethod()
        elif self.__initMode == 'kmeans++':
            self.__plusplusMethod()
        else:
            raise AttributeError

    # ------------------------------------------------------------------------------------------------------------------

    def __forgyMethod(self):
        """
        Initialization method:
        Randomly choose k observations from the data using a uniform random distribution.
        :return:
        """
        for ii in range(self.__k):
            self.__clusterMeans[ii] = (self.__obs[random.randint(0, self.__n - 1)])

    # ------------------------------------------------------------------------------------------------------------------

    def __uniformMethod(self):
        """
        Initialization method:
        Randomly assign each observation to one of the k clusters using uniform random distribution
        and compute the centroids of each cluster.
        :return:
        """
        for i in range(self.__n):
            self.__labelMap[i] = random.randint(0, self.__k - 1)

        self.__update()

    # ------------------------------------------------------------------------------------------------------------------

    def __randomMethod(self):
        """
        Initialization method:
        Randomly choose k observations from the data by estimating the mean and standard deviation of the data and
        using the gaussian random distribution.
        :return:
        """
        mean = np.mean(self.__obs, axis=0)
        std = np.std(self.__obs, axis=0)

        for ii in range(self.__k):
            self.__clusterMeans[ii] = np.random.normal(mean, std)

    # ------------------------------------------------------------------------------------------------------------------

    def __plusplusMethod(self):
        """
        Initialization method:
        Chooses k observations by computing probabilities for each observation and using a weighted random distribution.
        Algorithm: <https://en.wikipedia.org/wiki/K-means%2B%2B>. This method should accelerate the algorithm by finding
        the appropriate clusters right at the beginning and hence should make it more robust.
        :return:
        """
        # 1) choose random center out of data
        self.__clusterMeans[0] = (random.choice(self.__obs))

        maxValue = np.max(self.__obs) + 1
        probs = np.array([maxValue for _ in range(self.__n)])

        for i in range(1, self.__k):
            probs.fill(maxValue)
            # compute new probabilities, choose min of all distances
            for j in range(0, i):
                dists = similarityMeasurement(self.__obs, self.__clusterMeans[j], self.__compare)
                # collect minimum squared distances to cluster centroids
                probs = np.minimum(probs, dists)

            # sum all squared distances
            sumProbs = np.float(np.sum(probs))

            if sumProbs != 0:
                probs /= sumProbs
                # 3) choose new center based on probabilities
                self.__clusterMeans[i] = (self.__obs[weightedChoice(probs)])
            else:
                print('ERROR: cannot find enough cluster centroids for given k = ' + str(self.__k))

    # ------------------------------------------------------------------------------------------------------------------

    def getClusterMean(self, num):
        """
        Returns the centroid of the cluster with index num.
        :param num:
        :return:
        """
        if num >= self.__k:
            return None
        else:
            return self.__clusterMeans[num]

    # ------------------------------------------------------------------------------------------------------------------

    def getClusterOfElement(self, index):
        """
        :param index: number of element in observation array
        :return: cluster id of observation with given index.
        """
        if index >= self.__n:
            return None
        else:
            return self.__labelMap[index]

    # ------------------------------------------------------------------------------------------------------------------

    def printClusters(self):
        """
        Print the cluster centroids and the labels.
        :return:
        """
        print('Centroids: ' + str(self.__centroids) + ' | Labels: ' + str(self.__labels))

    # ------------------------------------------------------------------------------------------------------------------

    def __assignment(self):
        """
        Assignment step:
        Compute distance of current observation to each cluster centroid and move gene to the nearest cluster.
        :return:
        """
        for i in range(self.__n):
            value = self.__obs[i]

            # compute squared distances to each mean
            dists = similarityMeasurement(self.__clusterMeans, value, self.__compare)
            # nearest cluster
            nearestID = np.argmin(dists)

            if self.__labelMap[i] != nearestID:
                self.__changed = True
                self.__labelMap[i] = nearestID

    # ------------------------------------------------------------------------------------------------------------------

    def __update(self):
        """
        Update step:
        Compute the new centroids of each cluster after the assignment.
        :return:
        """
        self.__clusterMeans.fill(0)
        self.__clusterNums.fill(0)

        self.__clusterLabels = [[] for _ in range(self.__k)]

        for ii in range(self.__n):
            clusterID = self.__labelMap[ii]
            self.__clusterLabels[clusterID].append(ii)
            self.__clusterNums[clusterID] += 1

        for ii in range(self.__k):
            self.__clusterMeans[ii] = np.mean(self.__obs[self.__clusterLabels[ii]], axis=0)

    # ------------------------------------------------------------------------------------------------------------------

    def __end(self):
        """
        Writes the results to the corresponding member variables.
        :return:
        """
        # returned values | have to be reinitialized in case of sequential running
        # centroids
        self.__centroids = np.array([self.__obs[0] for _ in range(self.__k)], dtype=np.float)
        # labels of observations
        self.__labels = np.array([0 for _ in range(self.__n)], dtype=np.int)
        # distances between centroids
        # self.__centroidDistMat = np.zeros((self.__k, self.__k))

        # we do not use OrderedDict here, so obtain dict.values and fill array manually
        for index in range(self.__n):
            clusterID = self.__labelMap[index]
            self.__labels[index] = clusterID

        # collect centroids
        for ii in range(self.__k):
            # self.__centroids.append(self.__clusterMeans[ii].tolist())
            self.__centroids[ii] = self.__clusterMeans[ii]

        # compute distances between each centroids
        # for ii in range(self.__k - 1):
        #     # compute indices of other clusters
        #     jj = range(ii + 1, self.__k)
        #     # select matrix of cluster centroids
        #     centroidMat = self.__centroids[jj]
        #     distances = np.sqrt(self.__compare(centroidMat, self.__centroids[ii]))
        #     self.__centroidDistMat[ii, jj] = distances
        #     self.__centroidDistMat[jj, ii] = distances

    # ------------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        Runs the algorithm of k-means, using the initialization method and the assignment/update step.
        Conducts at most iters iterations and terminates if this number is exceeded or no observations
        was moved to another cluster.
        :return:
        """
        # 1) init algorithm by choosing cluster centroids
        self.__init()

        MAX_ITERS = self.__iters
        counter = 0
        # 2) run clustering
        while self.__changed and counter < MAX_ITERS:
            self.__changed = False

            self.__assignment()
            self.__update()

            counter += 1

        self.numIters = counter

        # write results to the class members
        self.__end()
        return self.__centroids.tolist(), self.__labels.tolist(), self.__clusterLabels
        #, self.__centroidDistMat.tolist()

    # ------------------------------------------------------------------------------------------------------------------

    # def getDistsPerCentroid(self):
    #     """
    #     Compute the distances between observations belonging to one cluster and the corresponding cluster centroid.
    #     Cluster labels are sorted in ascending order using their distances
    #     :return: array of distance arrays for each cluster and ordered labels
    #     """
    #
    #     # labels per centroid
    #     # self.__clusterLabels = [[] for _ in range(self.__k)]
    #     # distances of obs to their cluster
    #     self.__centroidDists = [[] for _ in range(self.__k)]
    #
    #     for ii in range(self.__k):
    #         self.__clusterLabels[ii] = np.array(self.__clusterLabels[ii], dtype=np.int)
    #
    #     # compute euclidean distances of values to cluster mean
    #     for ii in range(self.__k):
    #         mean = self.__clusterMeans[ii]
    #         obs = self.__obs[self.__clusterLabels[ii]]
    #         dists = similarityMeasurement(obs, mean, self.__compare).tolist()
    #         self.__centroidDists[ii] = dists
    #
    #         # sort indices in ascending order using the distances
    #         indices = range(len(dists))
    #         indices.sort(key=dists.__getitem__)
    #         self.__clusterLabels[ii] = self.__clusterLabels[ii][indices].tolist()
    #         self.__centroidDists[ii].sort()
    #
    #     return self.__clusterLabels, self.__centroidDists

########################################################################################################################

def _plugin_initialize():
  """
  optional initialization method of this module, will be called once
  :return:
  """
  pass

# ----------------------------------------------------------------------------------------------------------------------

def create(data, k, initMethod):
  """
  by convention contain a factory called create returning the extension implementation
  :return:
  """
  return KMeans(data, k, initMethod)

########################################################################################################################

from timeit import default_timer as timer
from scipy.cluster.vq import kmeans2, kmeans

"""
This is for testing the algorithm and comparing the resuls between this and scipy's algorithm
"""
if __name__ == '__main__':
    from datetime import datetime
    #np.random.seed(datetime.now())
    # data = np.array([[1,2,3],[5,4,5],[3,2,2],[8,8,7],[9,6,7],[2,3,4]])
    data = np.array([1,1.1,5,8,5.2,8.3])

    # data = np.array([np.random.rand(2) * 5 for _ in range(10)])
    k = 3

    timeMine = 0
    timeTheirs = 0
    n = 10

    for i in range(10):
        s1 = timer()
        kMeansPlus = KMeans(data, k, 'kmeans++', 10)
        result1 = kMeansPlus.run()
        #print(result)
        e1 = timer()
        # labels = kMeansPlus.getDistsPerCentroid()
        # l, d = computeClusterDistances(data, labels[0])

        s2 = timer()
        result2 = kmeans2(data, k)
        e2 = timer()

        timeMine += e1 - s1
        timeTheirs += e2 - s2

    print(result1)
    print(result2)
    print('mine: {}'.format(timeMine / n))
    print('theirs: {}'.format(timeTheirs / n))
