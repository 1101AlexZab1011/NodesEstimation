from typing import *

import numpy as np
import networkx as nx
import mne
from nodestimation.processing.connectivity import pearson, phase_locking_value
from nodestimation.processing.timewindow import mean_across_tw
from nodestimation.project.annotations import NodeFeatures


class Node(object):

    def __init__(
            self,
            label: mne.label.Label,
            features: NodeFeatures,
            center_coordinates: np.ndarray,
            type: Optional[str] = None
    ):
        self.features = features
        self.label = label
        self.center_coordinates = center_coordinates
        self.type = type

    def __str__(self):
        return 'Node for {}, {}'.format(self.label, self.type)

    @property
    def label(self):
        return self

    @property
    def center_coordinates(self):
        return self

    @property
    def features(self):
        return self

    @property
    def type(self):
        return self

    @label.setter
    def label(self, label: mne.label.Label):
        self._label = label

    @label.getter
    def label(self):
        return self._label

    @center_coordinates.setter
    def center_coordinates(self, coordinates: np.ndarray):

        if coordinates.shape[0] != 3:
            raise ValueError('Coordinates must have shape (3, ) but given shape is {}'.format(coordinates.shape))

        self._center_coordinates = coordinates

    @center_coordinates.getter
    def center_coordinates(self):
        return self._center_coordinates

    @features.setter
    def features(
            self,
            features: NodeFeatures
    ):
        self._features = features

    @features.getter
    def features(self):
        return self._features

    @type.setter
    def type(self, type: str):
        self._type = type

    @type.getter
    def type(self):
        return self._type


def eigencentrality(matrix: np.ndarray) -> np.ndarray:
    # computes eigencentrality for a square matrix

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
