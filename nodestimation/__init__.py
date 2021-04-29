from typing import *

import numpy as np
import networkx as nx
import mne
from nodestimation.processing.connectivity import pearson, phase_locking_value
from nodestimation.processing.timewindow import mean_across_tw
from nodestimation.project.annotations import NodeFeatures


class Node(object):
    """Brain unit representing one separated area of the brain

        :param label: label related to node
        :type label: |mne.Label|_
        :param features: dictionary representing measure value according to frequency
        :type features: *look for NodeFeatures in* :mod:`nodestimation.project.annotations`
        :param center_coordinates: x, y, and z coordinates of node position (in `mni coordinates <https://neuroimage.usc.edu/brainstorm/CoordinateSystems#MNI_coordinates>`_)
        :type center_coordinates: |inp.ndarray|_
        :param type: any information allowing distinguish one node (or group of nodes) from others
        :type type: str, optional
        :raises ValueError: if center_coordinates have `shape <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html>`_ other than (3,)

        .. _mne.Label: https://mne.tools/stable/generated/mne.Label.html?highlight=label#mne.Label
        .. _inp.ndarray:
        .. _np.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
        .. _idict: https://docs.python.org/3/library/stdtypes.html#dict
        .. _ifloat: https://docs.python.org/3/library/functions.html#float
        .. _istr: https://docs.python.org/3/library/stdtypes.html#str

        .. |mne.Label| replace:: *mne.Label*
        .. |inp.ndarray| replace:: *np.ndarray*
        .. |idict| replace:: *dict*

    """

    def __init__(
            self,
            label: mne.label.Label,
            features: NodeFeatures,
            center_coordinates: np.ndarray,
            type: Optional[str] = None
    ):
        """Constructor method
        """
        self.features = features
        self.label = label
        self.center_coordinates = center_coordinates
        self.type = type

    def __str__(self):
        """String representation
        """
        return 'Node for {}, {}'.format(self.label, self.type)

    @property
    def label(self):
        """`label <https://mne.tools/stable/generated/mne.Label.html?highlight=label#mne.Label>`_ related to node
        """

        return self

    @property
    def center_coordinates(self):
        """x, y, and z coordinates of node position (in `mni coordinates <https://neuroimage.usc.edu/brainstorm/CoordinateSystems#MNI_coordinates>`_)
        """

        return self

    @property
    def features(self):
        """dictionary representing measure value according to frequency
        """

        return self

    @property
    def type(self):
        """any information allowing distinguish one node (or group of nodes) from others
        """

        return self

    @label.setter
    def label(self, label: mne.label.Label):
        """setter for label"""

        self._label = label

    @label.getter
    def label(self):
        """`label <https://mne.tools/stable/generated/mne.Label.html?highlight=label#mne.Label>`_ related to node
        """

        return self._label

    @center_coordinates.setter
    def center_coordinates(self, coordinates: np.ndarray):
        """setter for center_coordinates"""

        if coordinates.shape[0] != 3:
            raise ValueError('Coordinates must have shape (3, ) but given shape is {}'.format(coordinates.shape))

        self._center_coordinates = coordinates

    @center_coordinates.getter
    def center_coordinates(self):
        """x, y, and z coordinates of node position (in `mni coordinates <https://neuroimage.usc.edu/brainstorm/CoordinateSystems#MNI_coordinates>`_)
        """

        return self._center_coordinates

    @features.setter
    def features(
            self,
            features: NodeFeatures
    ):
        """setter for features
        """

        self._features = features

    @features.getter
    def features(self):
        """dictionary representing measure value according to frequency
        """

        return self._features

    @type.setter
    def type(self, type: str):
        """setter for type
        """

        self._type = type

    @type.getter
    def type(self):
        """any information allowing distinguish one node (or group of nodes) from others
        """

        return self._type


def centrality(matrix: np.ndarray, centrality_metric: Callable, **kwargs) -> np.ndarray:
    """computes centrality for a square matrix with specified function

        :param matrix: a squared matrix for centrality computations
        :type matrix: |inp.ndarray|_
        :param centrality_metric: function to compute centrality
        :type: |icallable|_
        :return: vector with the same size as one row of given containing centrality value for each element
        :rtype: np.ndarray_
        :raises ValueError: if matrix have `shape <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html>`_ other than (:, :)

        .. _icallable: https://docs.python.org/3/library/typing.html#typing.Callable
        .. _np.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

        .. |icallable| replace:: *callable*
    """

    if len(matrix.shape) == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Can not compute centrality for non-square matrix')

        out = list()

        G = nx.from_numpy_matrix(matrix)
        centrality = centrality_metric(G, **kwargs)

        for node in centrality:
            out.append(centrality[node])

        return np.array(out)

    else:
        raise ValueError('Can work with two dimensions only')


def degree_centrality(matrix: np.ndarray) -> np.ndarray:
    """computes centrality for a square matrix with specified function

        :param matrix: a squared matrix for nodes degree computations
        :type matrix: |inp.ndarray|_
        :param centrality_metric: function to compute nodes degree
        :type: |icallable|_
        :return: vector with the same size as one row of given containing node degree for each element
        :rtype: np.ndarray_
        :raises ValueError: if matrix have `shape <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html>`_ other than (:, :)

        .. _icallable: https://docs.python.org/3/library/typing.html#typing.Callable
        .. _np.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

        .. |icallable| replace:: *callable*
    """

    if len(matrix.shape) == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Can not compute centrality for non-square matrix')

        out = list()

        G = nx.from_numpy_matrix(matrix)
        centrality = G.degree(weight='weight')

        for node in centrality:
            out.append(node[1])

        return np.array(out)

    else:
        raise ValueError('Can work with two dimensions only')


def eigencentrality(matrix: np.ndarray) -> np.ndarray:
    """Computes eigencentrality for a square matrix

        :param matrix: a squared matrix for eigencentrality computations
        :type matrix: |inp.ndarray|_
        :return: vector with the same size as one row of given containing eigencentrality value for each element
        :rtype: np.ndarray_
    """

    return centrality(matrix, nx.eigenvector_centrality_numpy, weight='weight')


def closeness_centrality(matrix: np.ndarray) -> np.ndarray:
    """Computes closeness centrality for a square matrix

        :param matrix: a squared matrix for closeness centrality computations
        :type matrix: |inp.ndarray|_
        :return: vector with the same size as one row of given containing closeness centrality value for each element
        :rtype: np.ndarray_
    """

    return centrality(matrix, nx.closeness_centrality, distance='weight')


def betweenness_centrality(matrix: np.ndarray) -> np.ndarray:
    """Computes betweenness centrality for a square matrix

        :param matrix: a squared matrix for betweenness centrality computations
        :type matrix: |inp.ndarray|_
        :return: vector with the same size as one row of given containing betweenness centrality value for each element
        :rtype: np.ndarray_
    """

    return centrality(matrix, nx.betweenness_centrality, weight='weight')


def katz_centrality(matrix: np.ndarray) -> np.ndarray:
    """Computes katz centrality for a square matrix

        :param matrix: a squared matrix for katz centrality computations
        :type matrix: |inp.ndarray|_
        :return: vector with the same size as one row of given containing katz centrality value for each element
        :rtype: np.ndarray_
    """

    return centrality(matrix, nx.katz_centrality, weight='weight')


def information_centrality(matrix: np.ndarray) -> np.ndarray:
    """Computes information centrality for a square matrix

        :param matrix: a squared matrix for information centrality computations
        :type matrix: |inp.ndarray|_
        :return: vector with the same size as one row of given containing information centrality value for each element
        :rtype: np.ndarray_
    """

    return centrality(matrix, nx.information_centrality, weight='weight')


def harmonic_centrality(matrix: np.ndarray) -> np.ndarray:
    """Computes harmonic centrality for a square matrix

        :param matrix: a squared matrix for harmonic centrality computations
        :type matrix: |inp.ndarray|_
        :return: vector with the same size as one row of given containing harmonic centrality value for each element
        :rtype: np.ndarray_
    """

    return centrality(matrix, nx.harmonic_centrality, distance='weight')
