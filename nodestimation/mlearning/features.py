import numpy as np
from scipy import interpolate


def prepare_connectivity(label_names, connectivity):

    out = connectivity

    for freq_band in connectivity.keys():
        for method in connectivity[freq_band].keys():
            conmat = connectivity[freq_band][method]['con'][:, :, 0]
            conmat_full = conmat + conmat.T
            out[freq_band][method] = conmat_full
    return {
        freq_band: {
            method: {
                label: row
                for label, row in zip(label_names, out[freq_band][method])
            } for method in connectivity[freq_band]
        } for freq_band in connectivity
    }


def prepare_psd(label_names, psd):
    return {
        freq_band: {
            label_name: psd[freq_band][label_name]['psd'].mean(axis=0)
            for label_name in label_names
        } for freq_band in psd.keys()
    }

def prepare_data(subjects):
    subjects_copy = subjects
    for subject in subjects_copy:
        for node in subject.nodes:
            for freq_band in node.features:
                node.features[freq_band]['psd'] = iterp_for_psd(node.features[freq_band]['psd'], len(subject.nodes))


    subjects_data = {
        subject.name: np.array([[
                node.label.name,
                np.array([
                    np.array([
                        node.features[freq_band][method] for method in node.features[freq_band]
                    ]) for freq_band in node.features
                ]),
                node.type
            ] for node in subject.nodes
        ]) for subject in subjects_copy
    }

    del subjects_copy
    return {
        subject.name : {
            'labels': subjects_data[subject.name][:, 0],
            'X': subjects_data[subject.name][:, 1],
            'Y': [data == 'resected' for data in subjects_data[subject.name][:, 2]],
        } for subject in subjects
    }

def iterp_for_psd(psd, n_samples):
    scale = np.arange(psd.shape[0])
    f = interpolate.interp1d(scale, psd, kind='cubic')
    scale_new = np.arange(0, psd.shape[0]-1, (psd.shape[0]-1)/n_samples)
    return f(scale_new)
