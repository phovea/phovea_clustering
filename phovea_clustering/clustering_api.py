__author__ = 'Michael Kern'
__version__ = '0.0.1'
__email__ = 'kernm@in.tum.de'

########################################################################################################################
# libraries

# use flask library for server activities
from phovea_server import ns
# load services (that are executed by the server when certain website is called)
from clustering_service import *

# create new flask application for hosting namespace
app = ns.Namespace(__name__)


########################################################################################################################

@app.route('/kmeans/<k>/<initMethod>/<distance>/<datasetID>')
def kmeansClustering(k, initMethod, distance, datasetID):
  """
  Access k-means clustering plugin.
  :param k: number of clusters
  :param initMethod:  initialization method for initial clusters
  :param distance: distance measurement
  :param datasetID:  identifier of data set
  :return: jsonified output
  """
  try:
    data = loadData(datasetID)
    response = runKMeans(data, int(k), initMethod, distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

@app.route('/hierarchical/<k>/<method>/<distance>/<datasetID>')
def hierarchicalClustering(k, method, distance, datasetID):
  """
  Access hierarchical clustering plugin.
  :param k: number of desired clusters
  :param method: type of single linkage
  :param distance: distance measurement
  :param datasetID: identifier of data set
  :return: jsonified output
  """
  try:
    data = loadData(datasetID)
    response = runHierarchical(data, int(k), method, distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

@app.route('/affinity/<damping>/<factor>/<preference>/<distance>/<datasetID>')
def affinityPropagationClustering(damping, factor, preference, distance, datasetID):
  """
  Access affinity propagation clustering plugin.
  :param damping:
  :param factor:
  :param preference:
  :param distance: distance measurement
  :param datasetID:
  :return:
  """
  try:
    data = loadData(datasetID)
    response = runAffinityPropagation(data, float(damping), float(factor), preference, distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

@app.route('/fuzzy/<numClusters>/<m>/<threshold>/<distance>/<datasetID>')
def fuzzyClustering(numClusters, m, threshold, distance, datasetID):
  """
  :param numClusters:
  :param m:
  :param threshold:
  :param distance:
  :param datasetID:
  :return:
  """
  try:
    data = loadData(datasetID)
    response = runFuzzy(data, int(numClusters), float(m), float(threshold), distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

def loadAttribute(jsonData, attr):
  import json
  data = json.loads(jsonData)
  if attr in data:
    return data[attr]
  else:
    return None


########################################################################################################################

@app.route('/distances/<metric>/<datasetID>/<sorted>', methods=['POST'])
def getDistances(metric, datasetID, sorted):
  """
  Compute the distances of the current stratification values to its centroid.
  :param metric:
  :param datasetID:
  :return: distances and labels sorted in ascending order
  """
  data = loadData(datasetID)
  labels = []
  externLabels = None

  if 'group' in ns.request.values:
    labels = loadAttribute(ns.request.values['group'], 'labels')
    externLabels = loadAttribute(ns.request.values['group'], 'externLabels')
  else:
    return ''

  response = getClusterDistances(data, labels, metric, externLabels, sorted)
  return ns.jsonify(response)


########################################################################################################################

@app.route('/dendrogram/<numClusters>/<datasetID>', methods=['POST'])
def dendrogramClusters(numClusters, datasetID):
  data = loadData(datasetID)

  if 'group' in ns.request.values:
    dendrogram = loadAttribute(ns.request.values['group'], 'dendrogram')
  else:
    return ''

  response = getClustersFromDendrogram(data, dendrogram, int(numClusters))
  return ns.jsonify(response)


########################################################################################################################

def create():
  """
  Standard Caleydo convention for creating the service when server is initialized.
  :return: Returns implementation of this plugin with given name
  """
  return app
