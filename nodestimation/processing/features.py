from typing import *

import numpy as np
import pandas as pd
from scipy import interpolate

import nodestimation
from nodestimation import eigencentrality
from nodestimation.project import read_or_write
from nodestimation.project.annotations import Features, LabelsFeatures, SubjectTree


def prepare_features(label_names: List[str], features: Features) -> LabelsFeatures:
    # creates a dictionary of dictionaries with the following structure: { feature_name: { label_name: feature_value } }

    def prepare_spectral_connectivity(label_names: List[str], connectivity: np.ndarray) -> Dict[str, float]:
        conmat = connectivity[:, :, 0]
        conmat_full = conmat + conmat.T
        conmat_full = eigencentrality(conmat_full)
        return {
            label: row
            for label, row in zip(label_names, conmat_full)
        }

    def prepare_psd(label_names: List[str], psd: np.ndarray) -> Dict[str, float]:
        return {
            label: np.sum(row.mean(axis=0))
            for label, row in zip(label_names, psd)
        }

    def prepare_envelope(label_names: List[str], envelope: np.ndarray) -> Dict[str, float]:
        envelope = eigencentrality(envelope)
        return {
            label: row
            for label, row in zip(label_names, envelope)
        }

    def prepare_pearson(label_names: List[str], pearson: np.ndarray) -> Dict[str, float]:
        pearson = eigencentrality(pearson)
        return {
            label: row
            for label, row in zip(label_names, pearson)
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
            }[method](label_names, features[freq_band][method])
            for method in features[freq_band] if method != 'pearson' and method != 'envelope'
        } for freq_band in features
    }

    if 'time-domain' in features:
        upd = {'time-domain': {}}
        if 'pearson' in features['time-domain']:
            upd['time-domain'].update({'pearson': prepare_pearson(label_names, features['time-domain']['pearson'])})
        if 'envelope' in features['time-domain']:
            upd['time-domain'].update({'envelope': prepare_envelope(label_names, features['time-domain']['envelope'])})
        out.update(upd)

    return out


@read_or_write('dataset')
def prepare_data(nodes: List[nodestimation.Node], _subject_tree: SubjectTree = None, _conditions: str = None, _priority: Optional[int] = None) -> pd.DataFrame:
    # creates a pandas DataFrame of features values with features and frequencies as columns and labels as index

    columns = list()
    keys = list()
    values = list()
    for freq_band in nodes[0].features:
        for method in nodes[0].features[freq_band]:
            if freq_band != 'time-domain':
                columns.append(freq_band + '_' + method)
            else:
                columns.append(method)

    columns.append('resected')

    for node in nodes:
        keys.append(node.label.name)

    for node in nodes:
        row = list()
        for freq_band in node.features:
            for method in node.features[freq_band]:
                row.append(
                    node.features[freq_band][method]
                )
        row.append(node.type == 'resected')
        values.append(row)

    data = dict(zip(keys, values))

    return pd.DataFrame.from_dict(data, orient='index', columns=columns)


def iterp_for_psd(psd: np.ndarray, n_samples: int) -> np.ndarray:
    # resamples given psd

    scale = np.arange(psd.shape[0])
    f = interpolate.interp1d(scale, psd, kind='cubic')
    scale_new = np.arange(0, psd.shape[0] - 1, (psd.shape[0] - 1) / n_samples)
    return f(scale_new)
