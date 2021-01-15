import numpy as np
import scipy as sp
import mne
from nodestimation.parcellation import freesurf_dict
from nodestimation.connectivity import pearson, phase_locking_value
from nodestimation.timewindow import mean_across_tw


class Node(object):

    def __init__(self, data, strength=None, label=None, type=None, connections=None):
        self.strength = strength
        self.data = data
        self.connections = connections

        if not isinstance(label, mne.Label):
            raise ValueError('label must be an instance of mne.Label')

        self.label = label
        self.type = type

        if label is not None:

            try:
                self.nilearn_coordinates = freesurf_dict[label.name]

            except KeyError:
                print("Unexpected label: " + label.name)
                self.nilearn_coordinates = None

        else:

            self.nilearn_coordinates = None

    def set_strength(self, strength):
        self.strength = strength

    def set_data(self, data):
        self.data = data

    def set_connections(self, connections):
        self.connections = connections

    def set_label(self, label):
        self.label = label

        try:
            self.nilearn_coordinates = freesurf_dict[label.name]

        except KeyError:
            print("Unexpected label")
            self.coordinates = None

    def set_coordinates(self, coordinates):

        if coordinates.shape[1] != 3:
            raise ValueError('Coordinates must have shape (n, 3) but given shape is {}'.format(coordinates.shape))

        self.nilearn_coordinates = coordinates

    def set_type(self, type, mood='rename'):

        if mood == 'rename':
            self.type = type

        elif mood == 'add':
            self.type += '/' + type

        else:
            raise ValueError("Unknown action: ", mood)


def central_node(*args):
    sum_strength = 0
    x_weight = 0
    y_weight = 0
    z_weight = 0

    for node in args:
        sum_strength += node.strength
        x_weight += (node.strength * node.nilearn_coordinates[0])
        y_weight += (node.strength * node.nilearn_coordinates[1])
        y_weight += (node.strength * node.nilearn_coordinates[2])

    out = Node(strength=sum_strength, type='computed_local_center')
    out.set_coordinates(np.array([x_weight / sum_strength, y_weight / sum_strength, z_weight / sum_strength]))

    return out


def eigencentrality(matrix):
    # only the greatest eigenvalue results in the desired centrality measure [Newman et al]
    if len(matrix.shape) == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Can not compute centrality for non-square matrix')
        out = np.real(sp.linalg.eigvals(matrix))

        return out

    elif len(matrix.shape) == 3:

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Matrix shape must be: [n x n x m]')

        c = [sp.linalg.eigvals(matrix[:, :, i]) for i in range(matrix.shape[-1])]
        out = [np.mean(np.real(np.array(c).T[i])) for i in range(matrix.shape[0])]

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
