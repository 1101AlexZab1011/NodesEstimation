from typing import *

import numpy as np
import networkx as nx
import mne
from nodestimation.processing.connectivity import pearson, phase_locking_value
from nodestimation.processing.timewindow import mean_across_tw
from nodestimation.project.annotations import NodeFeatures

class Node(object):
    """
    Brain unit representing one separated area of the brain
    this is essentially an extension of `mne.Label <https://mne.tools/stable/generated/mne.Label.html?highlight=label#mne.Label>`_ class

    :param label: label related to node
    :type label: mne.Label_
    :param features: dictionary representing measure value according to frequency
    :type features: dict
    :param center_coordinates: x, y, and z coordinates of node position (in `mni coordinates <https://neuroimage.usc.edu/brainstorm/CoordinateSystems#MNI_coordinates>`_)
    :type center_coordinates: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param type: any information allowing distinguish one node (or group of nodes) from others
    :type type: str, optional
    :raises ValueError: if center_coordinates have `shape <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html>`_ other than (3,)

    .. _mne.Label: https://mne.tools/stable/generated/mne.Label.html?highlight=label#mne.Label
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


def eigencentrality(matrix: np.ndarray) -> np.ndarray:
    """Computes eigencentrality for a square matrix

        :param matrix: a squared matrix for eigencentrality computations
        :type matrix: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        :return: matrix with the same size as given containing eigencentrality value for each element
        :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        :raises ValueError: if matrix have `shape <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html>`_ other than (:,2)
    """

    if len(matrix.shape) == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Can not compute centrality for non-square matrix')

        out = list()

        G = nx.from_numpy_matrix(matrix)
        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')

        for node in centrality:
            out.append(centrality[node])

        return np.array(out)

    else:
        raise ValueError('Can work with two dimensions only')
