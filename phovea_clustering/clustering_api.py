########################################################################################################################
# libraries

# use flask library for server activities
from phovea_server import ns
# load services (that are executed by the server when certain website is called)
from clustering_service import get_cluster_distances, get_clusters_from_dendrogram, load_data, run_affinity_propagation, run_fuzzy, run_hierarchical, run_kmeans


__author__ = 'Michael Kern'
__version__ = '0.0.1'
__email__ = 'kernm@in.tum.de'

# create new flask application for hosting namespace
app = ns.Namespace(__name__)


########################################################################################################################

@app.route('/kmeans/<k>/<init_method>/<distance>/<dataset_id>')
def kmeans_clustering(k, init_method, distance, dataset_id):
  """
  Access k-means clustering plugin.
  :param k: number of clusters
  :param init_method:  initialization method for initial clusters
  :param distance: distance measurement
  :param dataset_id:  identifier of data set
  :return: jsonified output
  """
  try:
    data = load_data(dataset_id)
    response = run_kmeans(data, int(k), init_method, distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

@app.route('/hierarchical/<k>/<method>/<distance>/<dataset_id>')
def hierarchical_clustering(k, method, distance, dataset_id):
  """
  Access hierarchical clustering plugin.
  :param k: number of desired clusters
  :param method: type of single linkage
  :param distance: distance measurement
  :param dataset_id: identifier of data set
  :return: jsonified output
  """
  try:
    data = load_data(dataset_id)
    response = run_hierarchical(data, int(k), method, distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

@app.route('/affinity/<damping>/<factor>/<preference>/<distance>/<dataset_id>')
def affinity_propagation_clustering(damping, factor, preference, distance, dataset_id):
  """
  Access affinity propagation clustering plugin.
  :param damping:
  :param factor:
  :param preference:
  :param distance: distance measurement
  :param dataset_id:
  :return:
  """
  try:
    data = load_data(dataset_id)
    response = run_affinity_propagation(data, float(damping), float(factor), preference, distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

@app.route('/fuzzy/<num_clusters>/<m>/<threshold>/<distance>/<dataset_id>')
def fuzzy_clustering(num_clusters, m, threshold, distance, dataset_id):
  """
  :param num_clusters:
  :param m:
  :param threshold:
  :param distance:
  :param dataset_id:
  :return:
  """
  try:
    data = load_data(dataset_id)
    response = run_fuzzy(data, int(num_clusters), float(m), float(threshold), distance)
    return ns.jsonify(response)
  except:
    return ns.jsonify({})


########################################################################################################################

def load_atttribute(json_data, attr):
  import json
  data = json.loads(json_data)
  if attr in data:
    return data[attr]
  else:
    return None


########################################################################################################################

@app.route('/distances/<metric>/<dataset_id>/<sorted>', methods=['POST'])
def get_distances(metric, dataset_id, sorted):
  """
  Compute the distances of the current stratification values to its centroid.
  :param metric:
  :param dataset_id:
  :return: distances and labels sorted in ascending order
  """
  data = load_data(dataset_id)
  labels = []
  extern_labels = None

  if 'group' in ns.request.values:
    labels = load_atttribute(ns.request.values['group'], 'labels')
    extern_labels = load_atttribute(ns.request.values['group'], 'externLabels')
  else:
    return ''

  response = get_cluster_distances(data, labels, metric, extern_labels, sorted)
  return ns.jsonify(response)


########################################################################################################################

@app.route('/dendrogram/<num_clusters>/<dataset_id>', methods=['POST'])
def dendrogram_clusters(num_clusters, dataset_id):
  data = load_data(dataset_id)

  if 'group' in ns.request.values:
    dendrogram = load_atttribute(ns.request.values['group'], 'dendrogram')
  else:
    return ''

  response = get_clusters_from_dendrogram(data, dendrogram, int(num_clusters))
  return ns.jsonify(response)


########################################################################################################################

def create():
  """
  Standard Caleydo convention for creating the service when server is initialized.
  :return: Returns implementation of this plugin with given name
  """
  return app
