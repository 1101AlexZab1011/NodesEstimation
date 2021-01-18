import numpy as np


def prepare_connectivity(label_names, connectivity):

    out = dict()

    for freq_band in connectivity.keys():
        for method in connectivity[freq_band].keys():
            conmat = connectivity[freq_band][method]['con'][:, :, 0]
            conmat_full = conmat + conmat.T
            out.update({
                freq_band: {
                    method: {
                        label_name: conmat_full_row
                    for label_name, conmat_full_row
                        in zip(label_names, conmat_full)
                    } for method in connectivity[freq_band].keys()
                } for freq_band in connectivity.keys()
            })
    return out


def prepare_psd(label_names, psd):
    return {
        freq_band: {
            label_name: psd[freq_band][label_name]['psd'].mean(axis=0)
            for label_name in label_names
        } for freq_band in psd.keys()
    }
