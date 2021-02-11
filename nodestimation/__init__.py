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
            type: Union[str, None] = None
    ):
        self.features = features
        self.label = label
        self.center_coordinates = center_coordinates
        self.features = features
        self.type = type

    def set_label(self, label: mne.label.Label):
        self.label = label

    def set_coordinates(self, coordinates: np.ndarray):

        if coordinates.shape[1] != 3:
            raise ValueError('Coordinates must have shape (n, 3) but given shape is {}'.format(coordinates.shape))

        self.center_coordinates = coordinates

    def set_features(
            self,
            features: NodeFeatures
    ):
        self.features = features

    def set_type(self, type: str, mood: str = 'rename'):

        if mood == 'rename':
            self.type = type

        elif mood == 'add':
            self.type += '/' + type

        else:
            raise ValueError("Unknown action: ", mood)


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
