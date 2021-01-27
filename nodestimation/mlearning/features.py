import numpy as np
from scipy import interpolate
from nodestimation.node_estimate import eigencentrality


def prepare_features(label_names, features):

    def prepare_spectral_connectivity(label_names, connectivity):
        conmat = connectivity[:, :, 0]
        conmat_full = conmat + conmat.T
        conmat_full = eigencentrality(conmat_full)
        return {
            label: row
            for label, row in zip(label_names, conmat_full)
        }

    def prepare_psd(label_names, psd):
        return {
            label: np.sum(row.mean(axis=0))
            for label, row in zip(label_names, psd)
        }

    def prepare_envelope(label_names, envelope):
        envelope = eigencentrality(envelope)
        return {
            label: row
            for label, row in zip(label_names, envelope)
        }

    def prepare_pearson(label_names, pearson):
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
        subject.name: {
            'labels': subjects_data[subject.name][:, 0],
            'X': subjects_data[subject.name][:, 1],
            'Y': [data == 'resected' for data in subjects_data[subject.name][:, 2]],
        } for subject in subjects
    }


def iterp_for_psd(psd, n_samples):
    scale = np.arange(psd.shape[0])
    f = interpolate.interp1d(scale, psd, kind='cubic')
    scale_new = np.arange(0, psd.shape[0] - 1, (psd.shape[0] - 1) / n_samples)
    return f(scale_new)
