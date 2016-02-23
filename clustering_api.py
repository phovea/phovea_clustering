__author__ = 'Michael Kern'
__version__ = '0.0.1'
__email__ = 'kernm@in.tum.de'

########################################################################################################################
# libraries

# use flask library for server activities
import flask
# load services (that are executed by the server when certain website is called)
from clustering_service import *

# create new flask application for hosting namespace
app = flask.Flask(__name__)

########################################################################################################################

@app.route('/kmeans/<k>/<initMethod>/<datasetID>')
def kmeansClustering(k, initMethod, datasetID):
    """
    Access k-means clustering plugin.
    :param k: number of clusters
    :param initMethod:  initialization method for initial clusters
    :param datasetID:  identifier of data set
    :return: jsonified output
    """
    try:
        data = loadData(datasetID)
        response = runKMeans(data, int(k), initMethod)
        return flask.jsonify(response)
    except:
        return flask.jsonify({})

########################################################################################################################

@app.route('/hierarchical/<method>/<datasetID>')
def singleLinkageClustering(method, datasetID):
    """
    Access hierarchical clustering plugin.
    :param method: type of single linkage
    :param datasetID: indentifier of data set
    :return: jsonified output
    """
    data = loadData(datasetID)
    response = runHierarchical(data, method)
    return flask.jsonify(response)

########################################################################################################################

@app.route('/affinity/<damping>/<factor>/<preference>/<datasetID>')
def affinityPropagationClustering(damping, factor, preference, datasetID):
    """
    Access affinity propagation clustering plugin.
    :param damping:
    :param factor:
    :param preference:
    :param datasetID:
    :return:
    """
    try:
        data = loadData(datasetID)
        response = runAffinityPropagation(data, float(damping), float(factor), preference)
        return flask.jsonify(response)
    except:
        return flask.jsonify({})

########################################################################################################################

def loadAttribute(jsonData, attr):
    import json
    data = json.loads(jsonData)
    if attr in data:
        return data[attr]
    else:
        return None

@app.route('/distances/<datasetID>', methods=['POST'])
def getDistances(datasetID):
    """
    Compute the distances of the current stratification values to its centroid.
    :param datasetID:
    :return: distances and labels sorted in ascending order
    """
    data = loadData(datasetID)
    labels = []
    externLabels = None

    if 'group' in flask.request.values:
        labels = loadAttribute(flask.request.values['group'], 'labels')
        externLabels = loadAttribute(flask.request.values['group'], 'externLabels')
    else:
        return ''

    response = getClusterDistances(data, labels, externLabels)
    return flask.jsonify(response)

########################################################################################################################

def create():
  """
  Standard Caleydo convention for creating the service when server is initialized.
  :return: Returns implementation of this plugin with given name
  """
  return app
