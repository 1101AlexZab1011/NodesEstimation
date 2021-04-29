from typing import *

import networkx as nx
import numpy as np
import pandas as pd
from scipy import interpolate
import nodestimation as nd
from nodestimation.project import read_or_write
from nodestimation.project.annotations import Features, LabelsFeatures, SubjectTree, Graphs, Connectomes


def prepare_graphs(features: Features, label_names: List[str]) -> Graphs:
    """Computes nx.Graph_ for each metric of `required metrics <nodestimation.html#list-of-metrics>`_

        :param label_names: `label <https://mne.tools/dev/generated/mne.Label.html>`_ names
        :type label_names: |ilist|_ *of* |istr|_
        :param features: `features <nodestimation.learning.html#feature>`_ to compute
        :type features: *look for Features in* :mod:`nodestimation.project.annotations`
        :return: dictionary with metrics names to built graph
        :rtype: look for Graphs in :mod:`nodestimation.project.annotations`

        .. _nx.Graph: https://networkx.org/documentation/stable/reference/classes/graph.html#networkx.Graph
    """

    def prepare_spectral_connectivity_graph(connectivity: np.ndarray, label_names: List[str]) -> nx.Graph:
        conmat = connectivity[:, :, 0]
        conmat_full = conmat + conmat.T
        g = nx.from_numpy_matrix(conmat_full)
        mapping = {node: label_name for node, label_name in zip(g, label_names)}
        g = nx.relabel_nodes(g, mapping)
        return g

    def prepare_correlation_connectivity_graph(connectivity: np.ndarray, label_names: List[str]) -> nx.Graph:
        g = nx.from_numpy_matrix(connectivity)
        mapping = {node: label_name for node, label_name in zip(g, label_names)}
        g = nx.relabel_nodes(g, mapping)
        return g

    return {
        freq_band: {
            method: {
                'psd': None,
                'coh': prepare_spectral_connectivity_graph,
                'cohy': prepare_spectral_connectivity_graph,
                'imcoh': prepare_spectral_connectivity_graph,
                'plv': prepare_spectral_connectivity_graph,
                'ciplv': prepare_spectral_connectivity_graph,
                'ppc': prepare_spectral_connectivity_graph,
                'pli': prepare_spectral_connectivity_graph,
                'pli2_unbiased': prepare_spectral_connectivity_graph,
                'wpli': prepare_spectral_connectivity_graph,
                'wpli2_debiased': prepare_spectral_connectivity_graph,
                'pearson': prepare_correlation_connectivity_graph,
                'envelope': prepare_correlation_connectivity_graph
            }[method](features[freq_band][method], label_names)
            for method in features[freq_band]
        } for freq_band in features
    }


def prepare_connectomes(features: Features, label_names: List[str]) -> Connectomes:
    """Computes connectome matrix for each metric of `required metrics <nodestimation.html#list-of-metrics>`_

        :param label_names: `label <https://mne.tools/dev/generated/mne.Label.html>`_ names
        :type label_names: |ilist|_ *of* |istr|_
        :param features: `features <nodestimation.learning.html#feature>`_ to compute
        :type features: *look for Features in* :mod:`nodestimation.project.annotations`
        :return: dictionary with metrics names to built connectome
        :rtype: look for Connectomes in :mod:`nodestimation.project.annotations`

        .. _nx.Graph: https://networkx.org/documentation/stable/reference/classes/graph.html#networkx.Graph
    """

    def prepare_spectral_connectivity_connectome(connectivity: np.ndarray, label_names: List[str]) -> pd.DataFrame:
        conmat = connectivity[:, :, 0]
        conmat_full = conmat + conmat.T
        return pd.DataFrame(conmat_full, index=label_names, columns=label_names)

    def prepare_correlation_connectivity_connectome(connectivity: np.ndarray, label_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame(connectivity, index=label_names, columns=label_names)

    return {
        freq_band: {
            method: {
                'psd': None,
                'coh': prepare_spectral_connectivity_connectome,
                'cohy': prepare_spectral_connectivity_connectome,
                'imcoh': prepare_spectral_connectivity_connectome,
                'plv': prepare_spectral_connectivity_connectome,
                'ciplv': prepare_spectral_connectivity_connectome,
                'ppc': prepare_spectral_connectivity_connectome,
                'pli': prepare_spectral_connectivity_connectome,
                'pli2_unbiased': prepare_spectral_connectivity_connectome,
                'wpli': prepare_spectral_connectivity_connectome,
                'wpli2_debiased': prepare_spectral_connectivity_connectome,
                'pearson': prepare_correlation_connectivity_connectome,
                'envelope': prepare_correlation_connectivity_connectome
            }[method](features[freq_band][method], label_names)
            for method in features[freq_band]
        } for freq_band in features
    }


def prepare_features(label_names: List[str], features: Features, centrality_metrics: Union[str, List[str]]) -> LabelsFeatures:
    """Computes `required metrics <nodestimation.html#list-of-metrics>`_ for each `label <https://mne.tools/dev/generated/mne.Label.html>`_

    :param label_names: `label <https://mne.tools/dev/generated/mne.Label.html>`_ names
    :type label_names: |ilist|_ *of* |istr|_
    :param features: `features <nodestimation.learning.html#feature>`_ to compute
    :type features: *look for Features in* :mod:`nodestimation.project.annotations`
    :param centrality_metrics: `centrality metrics <nodestimation.html#centrality-metrics>`_ to compute, default "eigen"
    :type centrality_metrics: str, optional
    :return: dictionary with label names to computed features
    :rtype: look for LabelsFeatures in :mod:`nodestimation.project.annotations`
    """

    if not isinstance(centrality_metrics, list):
        centrality_metrics = [centrality_metrics]

    centrality = {
        centrality_metric: {
            'degree': nd.degree_centrality,
            'eigen': nd.eigencentrality,
            'close': nd.closeness_centrality,
            'between': nd.betweenness_centrality,
            'katz': nd.katz_centrality,
            'info': nd.information_centrality,
            'harmonic': nd.harmonic_centrality
        }[centrality_metric]
        for centrality_metric in centrality_metrics
    }

    def prepare_spectral_connectivity(label_names: List[str], connectivity: np.ndarray) -> Dict[str, float]:
        conmat = connectivity[:, :, 0]
        conmat_full = conmat + conmat.T
        conmat_full = {
            metric: centrality[metric](conmat_full)
            for metric in centrality
        }
        return {
            metric:
                {
                    label: row
                    for label, row in zip(label_names, conmat_full[metric])
                } for metric in centrality
        }

    def prepare_correlation_connectivity(label_names: List[str], connectivity: np.ndarray) -> Dict[str, float]:
        conmat_full = {
            metric: centrality[metric](connectivity)
            for metric in centrality
        }
        return {
            metric:
                {
                    label: row
                    for label, row in zip(label_names, conmat_full[metric])
                } for metric in centrality
        }

    def prepare_psd(label_names: List[str], psd: np.ndarray) -> Dict[str, float]:
        psd = {
            label: np.sum(row.mean(axis=0))
            for label, row in zip(label_names, psd)
        }
        return {
            metric: psd for metric in centrality
        }

    out = {
        freq_band: {
            method: {
                'psd': prepare_psd,
                'coh': prepare_spectral_connectivity,
                'cohy': prepare_spectral_connectivity,
                'imcoh': prepare_spectral_connectivity,
                'plv': prepare_spectral_connectivity,
                'ciplv': prepare_spectral_connectivity,
                'ppc': prepare_spectral_connectivity,
                'pli': prepare_spectral_connectivity,
                'pli2_unbiased': prepare_spectral_connectivity,
                'wpli': prepare_spectral_connectivity,
                'wpli2_debiased': prepare_spectral_connectivity,
                'pearson': prepare_correlation_connectivity,
                'envelope': prepare_correlation_connectivity
            }[method](label_names, features[freq_band][method])
            for method in features[freq_band]
        } for freq_band in features
    }

    return out


@read_or_write('dataset')
def prepare_data(
        nodes: List[nd.Node],
        centrality_metrics: Union[str, List[str]],
        _subject_tree: SubjectTree = None,
        _conditions: str = None,
        _priority: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """for each centrality metric creates  pd.DataFrame_ from :class:`nodestimation.Node` `features <nodestimation.learning.html#feature>`_

        :param nodes: nodes to take information
        :type nodes: :class:`nodestimation.Node`
        :param centrality_metrics: `centrality metrics <nodestimation.html#centrality-metrics>`_ to compute, default "eigen"
        :type centrality_metrics: |ilist|_ *of* |istr|_ *or* |istr|_, optional
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _conditions: output from :func:`nodestimation.project.conditions_unique_code`, default True
        :type _conditions: str, optional
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: dataset with patient`s information
        :rtype: list_ of pd.DataFrame_

        .. _ipd.DataFrame:
        .. _pd.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

        .. _float:
        .. _ifloat: https://docs.python.org/3/library/functions.html#float
        .. _list: https://docs.python.org/3/library/stdtypes.html#list
        .. _tuple:
        .. _ituple: https://docs.python.org/3/library/stdtypes.html#tuple
        .. _str:
        .. _istr: https://docs.python.org/3/library/stdtypes.html#str
        .. _dict:
        .. _idict: https://docs.python.org/3/library/stdtypes.html#dict

        .. |ifloat| replace:: *float*
        .. |ituple| replace:: *tuple*
        .. |istr| replace:: *str*
        .. |idict| replace:: *dict*
    """

    columns = list()
    keys = list()
    datasets = dict()

    for freq_band in nodes[0].features:
        for method in nodes[0].features[freq_band]:
            if freq_band != 'time-domain':
                columns.append(freq_band + '_' + method)
            else:
                columns.append(method)

    columns.append('resected')

    for node in nodes:
        keys.append(node.label.name)

    for centrality in centrality_metrics:
        values = list()
        for node in nodes:
            row = list()
            for freq_band in node.features:
                for method in node.features[freq_band]:
                    row.append(
                        node.features[freq_band][method][centrality]
                    )
            row.append(node.type == 'resected')
            values.append(row)

        data = dict(zip(keys, values))

        datasets.update({centrality: pd.DataFrame.from_dict(data, orient='index', columns=columns)})

    return datasets


def iterp_for_psd(psd: np.ndarray, n_samples: int) -> np.ndarray:
    """resamples given psd using `interpolation <https://en.wikipedia.org/wiki/Interpolation>`_

    :param psd: array with `power spectral destinies <https://en.wikipedia.org/wiki/Spectral_density>`_
    :type psd: |inp.ndarray|_
    :param n_samples: how much samples given psd should have after `interpolation <https://en.wikipedia.org/wiki/Interpolation>`_
    :type n_samples: int
    :return: interpolated psd
    :rtype: np.ndarray_
    """

    scale = np.arange(psd.shape[0])
    f = interpolate.interp1d(scale, psd, kind='cubic')
    scale_new = np.arange(0, psd.shape[0] - 1, (psd.shape[0] - 1) / n_samples)

    return f(scale_new)
