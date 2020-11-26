from .parcellation import freesurf_dict
import numpy as np
import mne


class Node(object):

    def __init__(self, strength, data, label=None, type=None):
        self.strength = strength
        self.data = data
        if not isinstance(label, mne.Label):
            raise ValueError('label must be an instance of mne.Label')
        self.label = label
        self.type = type

        if label is not None:

            try:
                self.nilearn_coordinates = freesurf_dict[label.name]

            except KeyError:
                print("Unexpected label")
                self.nilearn_coordinates = None
        else:

            self.nilearn_coordinates = None

    def set_strength(self, strength):
        self.strength = strength

    def set_data(self, data):
        self.data = data

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

    def set_type(self, type):

        self.type = type


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
    out.set_coordinates(np.array([x_weight/sum_strength, y_weight/sum_strength, z_weight/sum_strength]))

    return out
