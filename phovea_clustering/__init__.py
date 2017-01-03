###############################################################################
# Caleydo - Visualization for Molecular Biology - http://caleydo.org
# Copyright (c) The Caleydo Team. All rights reserved.
# Licensed under the new BSD license, available at http://caleydo.org/license
###############################################################################


def phovea(registry):
  """
  register extension points
  :param registry:
  """
  # generator-phovea:begin
  registry.append('clustering', 'caleydo-clustering-kmeans', 'phovea_clustering.clustering_kmeans', {})

  registry.append('clustering', 'caleydo-clustering-hierarchical', 'phovea_clustering.clustering_hierarchical', {})

  registry.append('clustering', 'caleydo-clustering-affinity', 'phovea_clustering.clustering_affinity', {})

  registry.append('clustering', 'caleydo-clustering-fuzzy', 'phovea_clustering.clustering_fuzzy', {})

  registry.append('namespace', 'caleydo-clustering', 'phovea_clustering.clustering_api', {
      'namespace': '/api/clustering'
  })
  # generator-phovea:end
  pass


def phovea_config():
  """
  :return: file pointer to config file
  """
  from os import path
  here = path.abspath(path.dirname(__file__))
  config_file = path.join(here, 'config.json')
  return config_file if path.exists(config_file) else None
