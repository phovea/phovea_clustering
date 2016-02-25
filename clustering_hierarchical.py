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
# fastest distance computation by scipy
import scipy.spatial as spt

# utility functions for clustering and creating the dendrogram trees
from clustering_util import BinaryNode, BinaryTree
from clustering_util import similarityMeasurement
from clustering_util import computeClusterInternDistances

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

        # check if dimension is 2D
        if self.__obs.ndim == 2:
            # obtain number of observations (rows)
            numGenes, _ = np.shape(self.__obs)
            self.__n = numGenes

        else:
            print("[Error]:\tdata / observations must be 2D. 1D observation arrays are not supported")
            raise AttributeError

        # distance measurement
        self.__distance = distance

        # distance / proximity matrix
        self.__d = []
        self.__computeProximityMatrix()
        # dictionary mapping the string id (i,j,k,...) of clusters to corresponding index in matrix
        self.__idMap = {}
        # inverse mapping of idMap --> returns the string id given a certain index
        self.__keyMap = {}
        # contains actual index of all clusters, old clusters are from [0, n - 1], new clusters have indices in range
        # [n, 2n - 1]
        self.__clusterMap = {}
        for ii in range(self.__n):
            self.__idMap[str(ii)] = ii
            self.__keyMap[ii] = str(ii)
            self.__clusterMap[str(ii)] = ii

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

    def __getCoefficients(self, clusterI, clusterJ):
        """
        Compute the coefficients for the Lance-Williams algorithm
        :param clusterI:
        :param clusterJ:
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
        # TODO! || clusterI - clusterJ || ** 2
        elif self.__method == 'average':
            nI = np.float(clusterI.count(',') + 1)
            nJ = np.float(clusterJ.count(',') + 1)
            sumN = nI + nJ
            return (nI / sumN), (nJ / sumN), 0, 0

        # TODO! ATTENTION! centroid method should compute the cluster centroids using the mean
        # TODO! || clusterI - clusterJ || ** 2
        elif self.__method == 'centroid':
            nI = np.float(clusterI.count(',') + 1)
            nJ = np.float(clusterJ.count(',') + 1)
            sumN = nI + nJ
            return (nI / sumN), (nJ / sumN), -((nI * nJ) / (sumN ** 2)), 0

        # TODO! Support ward method
        # TODO! (|clusterI| * |clusterJ|) / (|clusterI| + |clusterJ) * || clusterI - clusterJ || ** 2
        # elif self.__method == 'ward':
        #     nI = np.float(clusterI.count(',') + 1)
        #     nJ = np.float(clusterJ.count(',') + 1)
        #     nK = np.float(clusterK.count(',') + 1)
        #     sumN = nI + nJ + nK
        #     return (nI + nK) / sumN, (nJ + nK) / sumN, -nK / sumN, 0
        else:
            raise AttributeError

    # ------------------------------------------------------------------------------------------------------------------

    def __computeProximityMatrix(self):
        """
        Compute the proximity of each observation and store the results in a nxn matrix
        :return:
        """

        # create distance matrix of size n x n
        self.__d = np.zeros((self.__n, self.__n))

        # compute euclidean distance
        # TODO! implement generic distance functions
        # TODO! look for an alternative proximity analysis without computing all distances
        self.__d = similarityMeasurement(self.__obs, self.__distance)

        # get number of maximum value of float
        self.__maxValue = self.__d.max() + 1

        # fill diagonals with max value to exclude them from min dist process
        # TODO! operate only on upper triangle matrix of distance matrix
        np.fill_diagonal(self.__d, self.__maxValue)

        # print('\t-> finished.')

    # ------------------------------------------------------------------------------------------------------------------

    def __getMatrixMinimumIndices(self):
        """
        Searches for the minimum distance in the distance matrix
        :return: indices of both clusters having the smallest distance
        """
        minDist = self.__d.min()
        minList = np.argwhere(self.__d == minDist)

        minI, minJ = 0, 0

        # look for indices, where i < j
        # TODO! for the future --> use upper triangle matrix
        for ii in range(len(minList)):
            minI, minJ = minList[ii]
            if minI < minJ:
                break

        if minI == minJ:
            print("ERROR")

        return self.__keyMap[minI], self.__keyMap[minJ], minDist

    # ------------------------------------------------------------------------------------------------------------------

    def __deleteClusters(self, i, j):
        """
        Reorders and reduces the matrix to insert the new cluster formed of cluster i and j
        and its distance values, and removes the old clusters by cutting the last row.
        :param i: cluster index i
        :param j: cluster index j
        :return:
        """
        idI = self.__idMap[str(i)]
        idJ = self.__idMap[str(j)]

        minID = min(idI, idJ)
        maxID = max(idI, idJ)

        # now set column max ID to last column -> swap last and i column
        lastRow = self.__d[self.__n - 1]
        self.__d[maxID] = lastRow
        self.__d[:, maxID] = self.__d[:, (self.__n - 1)]

        # set key of last column (cluster) to column of the cluster with index maxID
        key = self.__keyMap[self.__n - 1]
        self.__idMap[key] = maxID
        self.__keyMap[maxID] = key

        # delete entries in id and key map --> not required anymore
        try:
            del self.__idMap[i]
            del self.__idMap[j]
            del self.__keyMap[self.__n - 1]
        except KeyError:
            print("\nERROR: Key {} not found in idMap".format(j))
            print("ERROR: Previous key: {} in idMap".format(i))
            print("Given keys: ")
            for key in self.__idMap:
                print(key)
            return

        # reduce dimension of matrix by one column and row
        self.__n -= 1
        self.__d = self.__d[:-1, :-1]

    # ------------------------------------------------------------------------------------------------------------------

    def __mergeClusters(self, i, j):
        """
        Merges cluster i and j, computes the new ID and distances of the newly formed cluster
        and stores required information
        :param i: cluster index i
        :param j: cluster index j
        :return:
        """
        idI = self.__idMap[str(i)]
        idJ = self.__idMap[str(j)]

        minID = min(idI, idJ)
        maxID = max(idI, idJ)

        # use Lance-Williams formula to compute linkages
        DKI = self.__d[:, minID]
        DKJ = self.__d[:, maxID]
        DIJ = self.__d[minID, maxID]
        distIJ = np.abs(DKI - DKJ)

        # compute coefficients
        ai, aj, b, y = self.__getCoefficients(i, j)

        newEntries = ai * DKI + aj * DKJ + b * DIJ + y * distIJ
        newEntries[minID] = self.__maxValue
        newEntries[maxID] = self.__maxValue

        # add new column and row
        self.__d[minID] = newEntries
        self.__d[:, minID] = newEntries

        idIJ = minID
        newKey = i + ',' + j
        self.__idMap[newKey] = idIJ
        self.__keyMap[idIJ] = newKey
        self.__clusterMap[newKey] = len(self.__clusterMap)

        # delete old clusters
        self.__deleteClusters(i, j)

        # count number of elements
        return newKey.count(',') + 1

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
        Z = np.array([[0 for _ in range(4)] for _ in range(runs)], dtype=np.float)

        while m < runs:
            m += 1

            i, j, distIJ = self.__getMatrixMinimumIndices()
            numIJ = self.__mergeClusters(i, j)

            clusterI, clusterJ = self.__clusterMap[i], self.__clusterMap[j]
            Z[m - 1] = [int(min(clusterI, clusterJ)), int(max(clusterI, clusterJ)), np.float(distIJ), int(numIJ)]

        # reset number n to length of first dimension (number of genes)
        self.__n, _ = np.shape(self.__obs)

        self.__tree = self.generateTree(Z)
        return Z.tolist()

    # ------------------------------------------------------------------------------------------------------------------

    def generateTree(self, linkageMatrix):
        """
        Computes the dendrogram tree for a given linkage matrix.
        :param linkageMatrix:
        :return:
        """
        self.__tree = None

        treeMap = {}
        numTrees = len(linkageMatrix)

        for ii in range(numTrees):
            entry = linkageMatrix[ii]
            currentID = self.__n + ii
            leftIndex, rightIndex, value, num = int(entry[1]), int(entry[0]), entry[2], int(entry[3])
            left = right = None

            if leftIndex < self.__n:
                left = BinaryNode(self.__obs[leftIndex].tolist(), leftIndex, 1, None, None)
            else:
                left = treeMap[leftIndex]

            if rightIndex < self.__n:
                right = BinaryNode(self.__obs[rightIndex].tolist(), rightIndex, 1, None, None)
            else:
                right = treeMap[rightIndex]

            if isinstance(left, BinaryNode) and isinstance(right, BinaryNode):
                treeMap[currentID] = BinaryTree(left, right, currentID, value)
            elif isinstance(left, BinaryNode):
                treeMap[currentID] = right.addNode(left, currentID, value)
                del treeMap[rightIndex]
            elif isinstance(right, BinaryNode):
                treeMap[currentID] = left.addNode(right, currentID, value)
                del treeMap[leftIndex]
            else:
                treeMap[currentID] = left.merge(right, currentID, value)
                del treeMap[rightIndex]
                del treeMap[leftIndex]

        self.__tree = treeMap[numTrees + self.__n - 1]
        return self.__tree

    # ------------------------------------------------------------------------------------------------------------------

    def getClusters(self, k):
        """
        First implementation to cut dendrogram tree automatically by choosing nodes having the greatest node values
        or rather distance to the other node / potential cluster
        :param k: number of desired clusters
        :return: centroids, sorted cluster labels and normal label list
        """
        clusterLabels = self.__tree.cutTreeByClusters(k)

        clusterCentroids = []
        labels = np.zeros(self.__n, dtype=np.int)
        clusterID = 0

        for ii in range(len(clusterLabels)):
            cluster = clusterLabels[ii]
            obs = self.__obs[cluster]
            clusterCentroids.append(np.mean(obs, axis=0).tolist())

            for id in cluster:
                labels[id] = clusterID

            # sort labels according to their distance
            clusterLabels[ii], _ = computeClusterInternDistances(self.__obs, cluster)

            clusterID += 1

        return clusterCentroids, clusterLabels, labels.tolist()

########################################################################################################################

def _plugin_initialize():
    """
    optional initialization method of this module, will be called once
    :return:
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------

def create(data, method):
    """
    by convention contain a factory called create returning the extension implementation
    :return:
    """
    return Hierarchical(data, method)

########################################################################################################################

from timeit import default_timer as timer
from scipy.cluster.hierarchy import linkage, leaves_list

if __name__ == '__main__':
    np.random.seed(200)
    # data = np.array([[1,2,3],[5,4,5],[3,2,2],[8,8,7],[9,6,7],[2,3,4]])

    timeMine = 0
    timeTheirs = 0


    n = 10

    for i in range(n):
        data = np.array([np.random.rand(6000) * 4 - 2 for _ in range(249)])
        # import time
        s1 = timer()
        hier = Hierarchical(data, 'complete')
        # s = time.time()
        linkageMatrix = hier.run()
        e1 = timer()
        # print(linkageMatrix)
        tree = hier.generateTree(linkageMatrix)
        # print(tree.getLeaves())
        # print(tree.jsonify())
        # print(hier.getClusters(3))


        s2 = timer()
        linkageMatrix2 = linkage(data, 'complete')
        # print(leaves_list(linkageMatrix2))
        e2 = timer()

        timeMine += e1 - s1
        timeTheirs += e2 - s2

    # print(linkageMatrix)
    # print(linkageMatrix2)
    print('mine: {}'.format(timeMine / n))
    print('theirs: {}'.format(timeTheirs / n))

