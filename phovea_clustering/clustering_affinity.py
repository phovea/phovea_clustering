__author__ = 'Michael Kern'
__version__ = '0.0.1'
__email__ = 'kernm@in.tum.de'

########################################################################################################################
# libraries

# module to load own configurations
import phovea_server.config
# request config if needed for the future
config = phovea_server.config.view('caleydo-clustering')

import numpy as np
from clustering_util import similarityMeasurementMatrix
from timeit import default_timer as timer

########################################################################################################################

class AffinityPropagation:
    """
    This is an implementation of the affinity propagation algorithm to cluster genomic data / matrices.
    Implementation details: <http://www.psi.toronto.edu/index.php?q=affinity%20propagation>.
    Matlab implementation: <http://www.psi.toronto.edu/affinitypropagation/software/apcluster.m>
    Returns the centroids and labels / stratification of each row belonging to one cluster.
    """

    def __init__(self, obs, damping=0.5, factor=1.0, prefMethod='minimum', distance='euclidean'):
        """
        Initializes the algorithm.
        :param obs: genomic data / matrix
        :param damping: controls update process to dampen oscillations
        :param factor: controls the preference value (influences number of clusters)
        :param prefMethod: all points are chosen equally with a given preference (median or minimum of similarity matrix)
        :return:
        """
        self.__n = np.shape(obs)[0]
        # observations, can be 1D array or 2D matrix with genes as rows and conditions as columns
        # remove all NaNs in data
        self.__obs = np.nan_to_num(obs)
        # variables influencing output of clustering algorithm
        self.__damping = damping
        self.__factor = factor
        self.__prevMethod = prefMethod

        # similarity matrix
        self.__S = np.zeros((self.__n, self.__n))
        # availability matrix
        self.__A = np.zeros((self.__n, self.__n))
        # responsibility matrix
        self.__R = np.zeros((self.__n, self.__n))

        self.minValue = np.finfo(np.float).min

        # self.__mx1 = np.full(self.__n, self.minValue)
        # self.__mx2 = np.full(self.__n, self.minValue)

        self.__idx = np.zeros(self.__n)

        # set similarity computation
        self.__distance = distance

        self.__computeSimilarity()

    # ------------------------------------------------------------------------------------------------------------------

    def __call__(self):
        """
        Caller function for server API.
        """
        return self.run()

    # ------------------------------------------------------------------------------------------------------------------

    def __computeSimilarity(self):
        """
        Compute the similarity matrix from the original observation matrix and set preference of each element.
        :return: Similarity matrix
        """
        # compute distance matrix containing the negative sq euclidean distances -|| xi - xj ||**2
        self.__S = -similarityMeasurementMatrix(self.__obs, self.__distance)

        # determine the preferences S(k,k) to control the output of clusters
        pref = 0
        # could be median or minimum
        if self.__prevMethod == 'median':
            pref = float(np.median(self.__S)) * self.__factor
        elif self.__prevMethod == 'minimum':
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
        maxIter = 200
        maxConvIter = 100

        # sum all decisions for exemplars per round
        decisionSum = np.zeros(self.__n)
        # collect decisions for one exemplar per iteration round
        decisionIter = np.zeros((maxConvIter, self.__n))
        # counter for decisions (= consider data element as exemplar in each algorithm iteration)
        decisionCounter = maxConvIter
        # indicates if algorithm has converged
        isConverged = False

        centroids = []
        it = 0
        clusterI = []

        # helpful variables (that do not need recomputation)
        indexDiag = np.arange(self.__n)
        indicesDiag = np.diag_indices_from(self.__R)
        newA = np.zeros((self.__n, self.__n))
        newR = np.zeros((self.__n, self.__n))

        for it in range(1, maxIter + 1):

            # ----------------------------------------------------------------------------------------------------------

            # compute responsibility matrix
            AS = self.__A + self.__S

            maxY = np.max(AS, axis=1)
            indexY = np.argmax(AS, axis=1)

            # set values of maxima to zero in AS matrix
            AS[indexDiag, indexY] = self.minValue

            # look for second maxima
            maxY2 = np.max(AS, axis=1)

            # perform responsibility update
            for ii in range(self.__n):
                # s(i, k) - max({ a(i, k') + s(i, k') })
                newR[ii] = self.__S[ii] - maxY[ii]

            # subtract second maximum from row -> column entry with maximum value
            newR[indexDiag, indexY] = self.__S[indexDiag, indexY] - maxY2[indexDiag]

            # dampen values
            # self.__R = self.__damping * self.__R + (1 - self.__damping) * newR
            self.__R *= self.__damping
            self.__R += (1 - self.__damping) * newR

            # ----------------------------------------------------------------------------------------------------------

            # compute availability matrix
            # cut out negative elements
            # TODO! slow because of copy operation
            Rp = np.maximum(self.__R, 0)

            # write back all diagonal elements als self representatives
            Rp[indicesDiag] = self.__R[indicesDiag]
            sumCols = np.sum(Rp, axis=0)

            # apply availability update
            newA[:,] = sumCols
            newA -= Rp
            # for ii in range(self.__n):
            #     # r(k, k) + sum(max(0, r(i',k))
            #     newA[:, ii] = sumCols[ii] - Rp[:, ii]

            diagA = np.diag(newA)
            # take minimum of all the values in A, cut out all values above zero
            # newA = np.minimum(newA, 0)
            newA[newA > 0] = 0
            newA[indicesDiag] = diagA[indexDiag]

            # dampen values
            # self.__A = self.__damping * self.__A + (1 - self.__damping) * newA
            self.__A *= self.__damping
            self.__A += (1 - self.__damping) * newA

            # ----------------------------------------------------------------------------------------------------------

            # find exemplars for new clusters
            # old version which is slower
            # E = self.__R + self.__A
            # diagE = np.diag(E)

            # take the diagonal elements of the create matrix E
            diagE = np.diag(self.__R) + np.diag(self.__A)

            # all elements > 0 are considered to be an appropriate exemplar for the dataset
            clusterI = np.argwhere(diagE > 0).flatten()

            # count the number of clusters
            numClusters = len(clusterI)

            # ----------------------------------------------------------------------------------------------------------

            decisionCounter += 1
            if decisionCounter >= maxConvIter:
                decisionCounter = 0

            # subtract outcome of previous iteration (< 100) from the total sum of the decisions
            decisionSum -= decisionIter[decisionCounter]

            decisionIter[decisionCounter].fill(0)
            decisionIter[decisionCounter][clusterI] = 1

            # compute sum of decisions for each element being a exemplar
            decisionSum += decisionIter[decisionCounter]

            # check for convergence
            if it >= maxConvIter or it >= maxIter:
                isConverged = True

                for ii in range(self.__n):
                    # if element is considered to be an exemplar in at least one iterations
                    # and total of decisions in the last 100 iterations is not 100 --> no convergence
                    if decisionSum[ii] != 0 and decisionSum[ii] != maxConvIter:
                        isConverged = False
                        break

                if isConverged and numClusters > 0:
                    break

        # --------------------------------------------------------------------------------------------------------------

        # obtain centroids
        centroids = self.__obs[clusterI]

        # find maximum columns in AS matrix to assign elements to clusters / exemplars
        # fill A with negative values
        self.__A.fill(self.minValue)
        # set values of clusters to zero (as we only want to regard these values
        self.__A[:, clusterI] = 0.0
        # fill diagonal of similarity matrix to zero (remove preferences)
        np.fill_diagonal(self.__S, 0.0)

        # compute AS matrix
        AS = self.__A + self.__S
        # since values are < 0, look for the maximum number in each row and return its column index
        self.__idx = np.argmax(AS, axis=1)

        clusterI = clusterI.tolist()
        clusterLabels = [[] for _ in range(numClusters)]

        # create labels per cluster
        for ii in range(self.__n):
            index = clusterI.index(self.__idx[ii])
            self.__idx[ii] = index
            clusterLabels[index].append(ii)

        # return sorted cluster labels (that's why we call compute cluster distances, might be redundant)
        # for ii in range(numClusters):
        #     clusterLabels[ii], _ = computeClusterInternDistances(self.__obs, clusterLabels[ii])

        # if isConverged:
        #     print('Algorithm has converged after {} iterations'.format(it))
        # else:
        #     print('Algorithm has not converged after 200 iterations')
        #
        # print('Number of detected clusters {}'.format(numClusters))
        # print('Centroids: {}'.format(centroids))

        return centroids.tolist(), self.__idx.tolist(), clusterLabels

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
    data = np.array([1,1.1,5,8,5.2,8.3])

    s = timer()
    aff = AffinityPropagation(data, 0.9, 1.0, 'median', 'euclidean')
    result = aff.run()
    e = timer()
    print(result)
    print('time elapsed: {}'.format(e - s))

