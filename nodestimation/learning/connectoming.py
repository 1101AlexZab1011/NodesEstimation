from typing import *

import mne
import numpy as np
import nodestimation as nd
import pandas as pd
from matplotlib import pyplot as plt
import nodestimation.learning.modification as lmd

from nodestimation.project.subject import Subject


class Connectome(object):

    def __init__(self, subject: Subject, freq: str, method: str):
        self._matrix = subject.connectomes[freq][method]
        self._nodes = subject.nodes
        self._info = {
            'freq': freq,
            'method': method,
            'subject_name': subject.name,
            'size': subject.connectomes[freq][method].shape,
            'regions': [node.label.name for node in subject.nodes],
            'resection': [node.label.name for node in subject.nodes if node.type == 'resected']
        }

    def __str__(self) -> str:
        return f'{self._info["subject_name"]}\'s connectome of size {self._matrix.shape}' \
               f' obtained with {self._info["method"]} at {self._info["freq"]}'

    @property
    def matrix(self) -> pd.DataFrame:
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        raise AttributeError('Impossible to change connectivity matrix')

    @property
    def nodes(self) -> '(self: Subject) -> list[nd.Node]':
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        raise AttributeError('Impossible to change connectome-related nodes')

    @property
    def info(self):
        return self._info

    def inverse(self) -> pd.DataFrame:
        regions = self.matrix.index.tolist()
        matrix = self.matrix.to_numpy().copy()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    matrix[i, j] = 1 / matrix[i, j]
        return pd.DataFrame(
            matrix,
            index=regions,
            columns=regions
        )

    def spatial_map(self) -> pd.DataFrame:
        def dist(node1: nd.Node, node2: nd.Node):
            coords1 = node1.center_coordinates
            coords2 = node2.center_coordinates
            return np.sqrt(
                (coords1[0] - coords2[0]) ** 2 +
                (coords1[1] - coords2[1]) ** 2 +
                (coords1[2] - coords2[2]) ** 2
            )

        label_names = list()
        matrix_distances = list()
        for node1 in self.nodes:
            label_names.append(node1.label.name)
            row_distances = list()
            for node2 in self.nodes:
                row_distances.append(dist(node1, node2))
            matrix_distances.append(
                np.array(
                    row_distances
                )
            )
        return pd.DataFrame(matrix_distances, index=label_names, columns=label_names)

    def plot(self, connectome: Optional[pd.DataFrame] = None, title: Optional[str] = None):

        if connectome is None:
            connectome = self.matrix.to_numpy()
        else:
            if connectome.index.tolist() != self.matrix.index.tolist():
                raise ValueError('Given connectome has unexpected parcellation')
            connectome = connectome.to_numpy()

        if title is None:
            title = f'All-to-All {self.info["method"]}-based ' \
                    f'connectivity at {self.info["freq"]} ' \
                    f'for {self.info["subject_name"]}'
        else:
            title = f'All-to-All {self.info["method"]}-based ' \
                    f'connectivity at {self.info["freq"]} ' \
                    f'for {self.info["subject_name"]}' \
                    f'\n({title})'

        labels = [node.label for node in self.nodes]
        label_names = [label.name for label in labels]
        lh_labels = [name for name in label_names if name.endswith('lh')]
        rh_labels = [name for name in label_names if name.endswith('rh')]

        label_ypos_lh = list()

        for name in lh_labels:
            idx = label_names.index(name)
            ypos = np.mean(labels[idx].pos[:, 1])
            label_ypos_lh.append(ypos)

        try:
            idx = label_names.index('Brain-Stem')

        except ValueError:
            pass

        else:
            ypos = np.mean(labels[idx].pos[:, 1])
            lh_labels.append('Brain-Stem')
            label_ypos_lh.append(ypos)

        lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

        rh_labels = [label[:-2] + 'rh' for label in lh_labels
                     if label != 'Brain-Stem' and label[:-2] + 'rh' in rh_labels]

        node_colors = [label.color for label in labels]

        node_order = lh_labels[::-1] + rh_labels

        node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90,
                                              group_boundaries=[0, len(label_names) // 2])

        fig = plt.figure(num=None, figsize=(25, 25), facecolor='black')
        mne.viz.plot_connectivity_circle(connectome, label_names, n_lines=300,
                                         node_angles=node_angles, node_colors=node_colors,
                                         title=title,
                                         padding=8, fontsize_title=35, fontsize_colorbar=25,
                                         fontsize_names=20, fig=fig
                                         )


def make_connectome(
        subject: Subject,
        freq: str,
        method: str,
        kind: Optional[str] = 'initial',
        threshold: Optional[float] = 1
) -> pd.DataFrame:
    def spatial_related(connectome: Connectome, state: str = 'initial') -> pd.DataFrame:
        sp_map = connectome.spatial_map()

        if state == 'initial':
            return connectome.matrix * sp_map
        elif state == 'inverse':
            return connectome.inverse() * sp_map
        else:
            raise ValueError(f'{state} connectome does not exist')

    def binary(connectome_matrix: pd.DataFrame, threshold: Optional[float] = 1) -> pd.DataFrame:
        return lmd.binarize(
            connectome_matrix,
            trigger=threshold * connectome_matrix.to_numpy().mean().mean()
        )

    def suppressed(connectome_matrix: pd.DataFrame, threshold: Optional[float] = 1) -> pd.DataFrame:
        return lmd.suppress(
            connectome_matrix,
            trigger=threshold * connectome_matrix.to_numpy().mean().mean(),
            optimal=0
        )

    def spatial_related_bin(connectome: Connectome, state: str = 'initial',
                            threshold: Optional[float] = 1) -> pd.DataFrame:
        sp_map = connectome.spatial_map()

        if state == 'initial':
            bin_con = binary(connectome.matrix, threshold)
            return sp_map * bin_con
        elif state == 'inverse':
            bin_con = binary(connectome.inverse(), threshold)
            return sp_map * bin_con
        else:
            raise ValueError(f'{state} connectome does not exist')

    def spatial_related_supp(connectome: Connectome, state: str = 'initial',
                             threshold: Optional[float] = 1) -> pd.DataFrame:
        sp_map = connectome.spatial_map()

        if state == 'initial':
            supp_con = suppressed(connectome.matrix, threshold)
            return sp_map * supp_con
        elif state == 'inverse':
            supp_con = suppressed(connectome.inverse(), threshold)
            return sp_map * supp_con
        else:
            raise ValueError(f'{state} connectome does not exist')

    return {
        'initial': Connectome(subject, freq, method).matrix,
        'binary': binary(Connectome(subject, freq, method).matrix, threshold=threshold),
        'suppressed': suppressed(Connectome(subject, freq, method).matrix, threshold=threshold),
        'inverse': Connectome(subject, freq, method).inverse(),
        'inverse_binary': binary(Connectome(subject, freq, method).inverse(), threshold=threshold),
        'inverse_suppressed': suppressed(Connectome(subject, freq, method).inverse(), threshold=threshold),
        'spatial': Connectome(subject, freq, method).spatial_map(),
        'initial&spatial': spatial_related(Connectome(subject, freq, method)),
        'inverse&spatial': spatial_related(Connectome(subject, freq, method), 'inverse'),
        'bin&spatial': spatial_related_bin(Connectome(subject, freq, method), threshold=threshold),
        'supp&spatial': spatial_related_supp(Connectome(subject, freq, method), threshold=threshold),
        'inverse-bin&spatial': spatial_related_bin(Connectome(subject, freq, method), 'inverse', threshold=threshold),
        'inverse-supp&spatial': spatial_related_supp(Connectome(subject, freq, method), 'inverse', threshold=threshold)
    }[kind]
