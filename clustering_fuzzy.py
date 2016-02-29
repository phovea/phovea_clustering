__author__ = 'Michael Kern'
__version__ = '0.0.3'
__email__ = 'kernm@in.tum.de'

########################################################################################################################
# libraries

# module to load own configurations
import caleydo_server.config
# request config if needed for the future
config = caleydo_server.config.view('caleydo-clustering')

# library to conduct matrix/vector calculus
import numpy as np

from clustering_util import euclideanDistance, computeClusterInternDistances

########################################################################################################################
# class definition

class Fuzzy(object):
    """
    Formulas: https://en.wikipedia.org/wiki/Fuzzy_clustering
    """

    def __init__(self, obs, numClusters, m=2, init=None, distance=euclideanDistance, error=0.0001):
        """

        :param obs:
        :param clusters:
        :param m:
        :return:
        """
        # observation
        self.__obs = np.nan_to_num(obs)

        self.__n = obs.shape[0]

        # fuzzifier value
        self.__m = m
        # number of clusters
        self.__c = numClusters

        # matrix u containing all the weights describing the degree of membership of each patient to the centroid
        if init is None:
            init = np.random.rand(self.__c, self.__n)

        self.__u = np.copy(init)

        # TODO! scikit normalizes the values at the beginning and at each step to [0;1]
        self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(np.sum(self.__u, axis=0))).astype(np.float64)

        # remove all zero values and set them to smallest possible value
        self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)
        # centroids
        self.__centroids = np.zeros(self.__c)
        # threshold for stopping criterion
        self.__error = error
        # distance function
        self.__distance = distance


    def __call__(self):
        """
        Caller function for server API
        :return:
        """
        return self.run()

    def computeCentroid(self):
        """

        :return:
        """
        # normalize data and eliminate zero values
        # uOld /= np.ones((self.__c, 1)).dot(np.atleast_2d(np.sum(uOld, axis=0)))
        # uOld = np.fmax(uOld, np.finfo(np.float64).eps)

        uM = self.__u ** self.__m

        sumDataWeights = np.dot(uM, self.__obs)
        m = self.__obs.shape[1]

        sumWeights = np.sum(uM, axis=1)
        # tile array (sum of weights repeated in every row)
        sumWeights = np.ones((m, 1)).dot(np.atleast_2d(sumWeights)).T
        # divide by total sum to get new centroids
        self.__centroids = sumDataWeights / sumWeights

    def computeCoefficients(self):
        """

        :return:
        """

        # TODO you can also use cdist of scipy.spatial.distance module
        distMat = np.zeros((self.__c, self.__n))

        for ii in range(self.__c):
            distMat[ii] = self.__distance(self.__obs, self.__centroids[ii])

        # set zero values to smallest values to prevent inf results
        distMat = np.fmax(distMat, np.finfo(np.float64).eps)

        # apply coefficient formula
        self.__u = distMat ** (-2.0 / (self.__m - 1))
        sumCoeffs = np.sum(self.__u, axis=0)
        self.__u /= np.ones((self.__c, 1)).dot(np.atleast_2d(sumCoeffs))
        self.__u = np.fmax(self.__u, np.finfo(np.float64).eps)

    def run(self):
        """

        :return:
        """
        MAX_ITER = 1000
        iter = 0

        while iter < MAX_ITER:
            # copy old weights matrix
            uOld = np.copy(self.__u)
            # compute centroids with given weights
            self.computeCentroid()
            # compute new coefficient matrix
            self.computeCoefficients()

            # compute the difference between the old and new matrix
            epsilon = np.linalg.norm(self.__u - uOld)
            # stop if difference (epsilon) is smaller than the user-defined threshold
            if epsilon < self.__error:
                break

        self.__end()

        return self.__centroids.tolist(), self.__clusterLabels#, self.__u.T.tolist()

    def __end(self):
        """

        :return:
        """
        # assign patient to clusters
        # transpose to get a (n, c) matrix
        u = self.__u.T

        self.__labels = np.zeros(self.__n, dtype=np.int)
        self.__clusterLabels = [[] for _ in range(self.__c)]

        maxProb = 1.0 / self.__c

        for ii in range(self.__n):
            clusterID = np.argmax(u[ii])
            self.__labels = clusterID
            self.__clusterLabels[clusterID].append(ii)

            # for jj in range(self.__c):
                # if u[ii][jj] >= maxProb:
                #   clusterID = jj
                #   self.__labels = clusterID
                #   self.__clusterLabels[clusterID].append(ii)

        for ii in range(self.__c):
            self.__clusterLabels[ii], _ = computeClusterInternDistances(self.__obs, self.__clusterLabels[ii])

########################################################################################################################

def _plugin_initialize():
    """
    optional initialization method of this module, will be called once
    :return:
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------

def create(data, numCluster, m):
    """
    by convention contain a factory called create returning the extension implementation
    :return:
    """
    return Fuzzy(data, numCluster, m)

########################################################################################################################

if __name__ == '__main__':

    data = np.array([[1,2,3],[5,4,5],[3,2,2],[8,8,7],[9,6,7],[2,3,4]])

    fuz = Fuzzy(data, 3, 1.1)
    print(fuz.run())
