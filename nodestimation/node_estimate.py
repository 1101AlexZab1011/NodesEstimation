import numpy as np
import scipy as sp
import mne
from nodestimation.connectivity import pearson, phase_locking_value
from nodestimation.timewindow import mean_across_tw


class Node(object):

    def __init__(self, label, features, nilearn_coordinates=None, type=None, ml_class=None):
        self.features = features

        if not isinstance(label, mne.Label):
            raise ValueError('label must be an instance of mne.Label')

        self.label = label
        self.nilearn_coordinates = nilearn_coordinates
        self.features = features
        self.type = type
        self.ml_class = ml_class

    def set_label(self, label):
        self.label = label

    def set_coordinates(self, coordinates):

        if coordinates.shape[1] != 3:
            raise ValueError('Coordinates must have shape (n, 3) but given shape is {}'.format(coordinates.shape))

        self.nilearn_coordinates = coordinates

    def set_features(self, features):
        self.features = features

    def set_type(self, type, mood='rename'):

        if mood == 'rename':
            self.type = type

        elif mood == 'add':
            self.type += '/' + type

        else:
            raise ValueError("Unknown action: ", mood)

    def classify_as(self, ml_class):
        self.ml_class = ml_class


def eigencentrality(matrix):
    # only the greatest eigenvalue results in the desired centrality measure [Newman et al]
    if len(matrix.shape) == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Can not compute centrality for non-square matrix')
        out = sp.linalg.eigvalsh(matrix)

        return out

    elif len(matrix.shape) == 3:

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Matrix shape must be: [n x n x m]')

        c = [sp.linalg.eigvalsh(matrix[:, :, i]) for i in range(matrix.shape[-1])]
        out = [np.mean(np.array(c).T[i]) for i in range(matrix.shape[0])]

        return np.array(out)

    else:
        raise ValueError('Can not work with dimension less than two and higher than four')


def nodes_strength(label_tc, method):
    if method == 'pearson':
        pearson_matrices = pearson(label_tc)
        pears_mean = mean_across_tw(pearson_matrices)
        n_strength = np.array([])

        for i in range(pears_mean.shape[0]):
            n_strength = np.append(n_strength, np.sum(pears_mean[i, :]))

        return n_strength, pears_mean

    elif method == 'plv':
        plv_matrices = phase_locking_value(label_tc)
        plv_mean = mean_across_tw(plv_matrices)
        centrality = eigencentrality(plv_mean)

        return centrality, plv_mean
