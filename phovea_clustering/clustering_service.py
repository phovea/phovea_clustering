__author__ = 'Michael Kern'
__version__ = '0.0.1'
__email__ = 'kernm@in.tum.de'

import numpy as np
from clustering_hierarchical import getClusters

########################################################################################################################

def loadData(datasetID):
    """
    Loads the genomic data with given identifier datasetID.
    :param datasetID: identifier
    :return: array of the genomic data
    """
    import phovea_server.dataset as dt
    # obtain Caleydo dataset from ID
    dataset = dt.get(datasetID)
    # choose loaded attribute and load raw data in numpy format
    # somehow hack to get a numpy array out of the data
    try:
        arr = np.array(list(dataset.asnumpy()))
    except:
        raise Exception
    return arr

########################################################################################################################

def loadPlugin(pluginID, *args, **kwargs):
    """
    Loads the clustering plugin with given arguments.
    :param pluginID: identifier of plugin
    :param *args: additional caller function arguments
    :param **kwargs: additional arguments
    :return: plugin
    """
    import phovea_server.plugin
    # obtain all plugins with 'pluginID' extension
    plugins = phovea_server.plugin.list('clustering')
    # choose plugin with given ID
    for plugin in plugins:
        if plugin.id == pluginID:
            # load the implementation of the plugin
            return plugin.load().factory(*args, **kwargs)

    raise NotImplementedError


########################################################################################################################

def runKMeans(data, k, initMethod, distance):
    """
    Runs the k-Means clustering algorithm given the loaded data set, the number of clusters k and the initialization
    method.
    :param data: observation matrix
    :param k: number of clusters
    :param initMethod: number of clusters
    :return: result of k-means
    """
    KMeans = loadPlugin('caleydo-clustering-kmeans', data, k, initMethod, distance)
    # and run the kmeans extension
    centroids, labels, clusterLabels = KMeans()
    # clusterLabels, clusterDists = KMeans.getDistsPerCentroid()

    return {'centroids': centroids, 'clusterLabels': clusterLabels}

########################################################################################################################

def runHierarchical(data, k, method, distance):
    """
    Runs the hierarchical clustering algorithm given the loaded data set and type of linkage method.
    :param data: observation matrix
    :param method: linkage method
    :return: linkage matrix / dendrogram of the algorithm
    """
    Hierarchical = loadPlugin('caleydo-clustering-hierarchical', data, method, distance)
    # and use the extension
    Hierarchical()
    # obtain k-number of clusters
    centroids, clusterLabels, labels = getClusters(k, data, Hierarchical.tree, False)

    return {'centroids': centroids, 'clusterLabels': clusterLabels, 'dendrogram': Hierarchical.tree.json()}
    # print('\t-> creating dendrogram tree...')
    # tree = Hierarchical.generateTree(linkage)
    # print('\t-> creating json string ...')
    # dendrogram = tree.jsonify()
    # print('\t-> finished.')

    # return {'dendrogram': dendrogram} --> if needed later

########################################################################################################################

def runAffinityPropagation(data, damping, factor, preference, distance):
    """
    Runs the affinity propagation algorithm given the loaded dataset, a damping value, a certain factor and
    a preference method.
    :param data:
    :param damping:
    :param factor:
    :param preference:
    :return:
    """
    Affinity = loadPlugin('caleydo-clustering-affinity', data, damping, factor, preference, distance)
    # use this extension
    centroids, labels, clusterLabels = Affinity()

    return {'centroids': centroids, 'clusterLabels': clusterLabels}

########################################################################################################################

def runFuzzy(data, numClusters, m, threshold, distance):
    Fuzzy = loadPlugin('caleydo-clustering-fuzzy', data, numClusters, m, threshold, distance)

    centroids, clusterLabels, partitionMatrix, maxProb = Fuzzy()

    return {'centroids': centroids, 'clusterLabels': clusterLabels, 'partitionMatrix': partitionMatrix,
            'maxProbability': maxProb}

########################################################################################################################

def getClusterDistances(data, labels, metric, externLabels = None, sorted = True):
    """
    Compute the cluster distances in a given data among certain rows (labels)
    :param data: genomic data
    :param labels: indices of rows
    :param metric: distance metric
    :param externLabels:
    :return: labels and distances values sorted in ascending order
    """
    from clustering_util import computeClusterInternDistances, computeClusterExternDistances
    distLabels, distValues = computeClusterInternDistances(data, labels, sorted, metric)

    if externLabels is not None:
        externDists = computeClusterExternDistances(data, distLabels, externLabels, metric)
        return {'labels': distLabels, 'distances': distValues, 'externDistances': externDists}
    else:
        return {'labels': distLabels, 'distances': distValues}

########################################################################################################################

def getClustersFromDendrogram(data, dendrogram, numClusters):
    """

    :param data:
    :param dendrogram:
    :param numClusters:
    :return:
    """

    centroids, clusterLabels, _ = getClusters(numClusters, data, dendrogram)
    return {'centroids': centroids, 'clusterLabels': clusterLabels}
