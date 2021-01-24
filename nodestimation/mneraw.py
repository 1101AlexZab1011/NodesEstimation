import os
import re
import mne
import nibabel
import hashlib
import numpy as np
import nilearn.image as image
from nodestimation.project.path import read_or_write
from nodestimation.node_estimate import Node
import nodestimation.project.path as path
from nodestimation.mlearning.features import \
    prepare_connectivity,\
    prepare_psd
from nodestimation.project.structures import\
    connectivity_computation_output_features,\
    subject_data_types
from nodestimation.project.subject import Subject


def notchfir(raw, lfreq, nfreq, hfreq):
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    raw_filtered = raw \
        .load_data() \
        .notch_filter(nfreq, meg_picks) \
        .filter(l_freq=lfreq, h_freq=hfreq)

    return raw_filtered


def artifacts_clean(raw):
    ica = mne.preprocessing.ICA(n_components=15, random_state=97)
    ica.fit(raw)
    ica.exclude = ica.find_bads_eog(raw)[0] + \
                  ica.find_bads_ecg(raw, method='correlation', threshold=3.0)[0]

    ica.apply(raw)

    del ica

    return raw


@read_or_write('raw', target='original', write_file=False)
def read_original_raw(path, _subject_tree=None, _conditions=None):
    return mne.io.read_raw_fif(path)


@read_or_write('raw', target='nepf')
def first_processing(raw, lfreq, nfreq, hfreq,
                     rfreq=None,
                     crop=None,
                     reconstruct=False,
                     meg=True,
                     eeg=True,
                     _subject_tree=None,
                     _conditions=None):
    out = raw.copy()

    if crop:
        if not isinstance(crop, list):
            out.crop(tmax=crop)
        elif isinstance(crop, list) \
                and len(crop) == 2:
            out.crop(tmin=crop[0], tmax=crop[1])
        else:
            raise ValueError('Crop range is incorrect: {}'.format(crop))

    if rfreq:
        out.resample(rfreq, npad='auto')

    out = notchfir(out, lfreq, nfreq, hfreq)
    if reconstruct:
        out = artifacts_clean(out)

    out.pick_types(meg=meg, eeg=eeg)

    return out


@read_or_write('bem')
def bem_computation(subject, subjects_dir, conductivity, _subject_tree=None, _conditions=None):
    model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
    return mne.make_bem_solution(model)


@read_or_write('src')
def src_computation(subject, subjects_dir, bem, _subject_tree=None, _conditions=None):
    labels_vol = ['Left-Amygdala',
                  'Left-Thalamus-Proper',
                  'Left-Cerebellum-Cortex',
                  'Brain-Stem',
                  'Right-Amygdala',
                  'Right-Thalamus-Proper',
                  'Right-Cerebellum-Cortex']
    fname_aseg = os.path.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    vol_src = mne.setup_volume_source_space(
        subject, mri=fname_aseg,
        pos=10.0, bem=bem,
        add_interpolator=True,
        volume_label=labels_vol,
        subjects_dir=subjects_dir
    )
    src = mne.setup_source_space(subject, spacing='ico5', add_dist='patch', subjects_dir=subjects_dir)
    return src + vol_src


@read_or_write('trans', target='original', write_file=False)
def read_original_trans(path, _subject_tree=None, _conditions=None):
    return mne.read_trans(path)


@read_or_write('fwd')
def forward_computation(raw, trans, src, bem, _subject_tree=None, _conditions=None):
    return mne.make_forward_solution(raw, trans=trans, src=src, bem=bem, meg=True, eeg=False,
                                     mindist=5.0, n_jobs=1, verbose=True)


@read_or_write('eve')
def events_computation(raw, time_points, ids, _subject_tree=None, _conditions=None):
    return np.array([[
        raw.first_samp + raw.time_as_index(time_point)[0],
        0,
        event_id

    ] for time_point, event_id in zip(time_points, ids)])


@read_or_write('epo')
def epochs_computation(raw, events, tmin, tmax, _subject_tree=None, _conditions=None):
    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax)


@read_or_write('cov')
def noise_covariance_computation(epochs, tmin, tmax, _subject_tree=None, _conditions=None):
    return mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='empirical')


@read_or_write('ave')
def evokeds_computation(epochs, _subject_tree=None, _conditions=None):
    return epochs.average()


@read_or_write('inv')
def inverse_computation(info, fwd, cov, _subject_tree=None, _conditions=None):
    return mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=None, fixed=False)


@read_or_write('stc')
def source_estimation(epochs, inv, lambda2, method, _subject_tree=None, _conditions=None):
    return mne.minimum_norm.apply_inverse_epochs(epochs,
                                                 inv,
                                                 lambda2,
                                                 method,
                                                 pick_ori=None
                                                 )


@read_or_write('coords')
def coordinates_computation(subject, subjects_dir, labels, _subject_tree=None, _conditions=None):
    vertexes = [mne.vertex_to_mni(
        label.vertices,
        hemis=0 if label.hemi == 'lh' else 1,
        subject=subject, subjects_dir=subjects_dir
    )for label in labels]
    return {label.name: np.mean(vertex, axis=0) for label, vertex in zip(labels, vertexes)}


@read_or_write('resec', target='original', write_file=False)
def read_original_resec(path, _subject_tree=None, _conditions=None):
    return nibabel.load(path)


@read_or_write('resec_mni')
def resection_area_computation(img, _subject_tree=None, _conditions=None):
    res = np.array(img.get_data().tolist())
    img_coordinates = list()
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            for k in range(res.shape[2]):
                if res[i, j, k] != 0:
                    img_coordinates.append(np.array([i, j, k]))
    img_coordinates = np.array(img_coordinates)
    mni_coordinates = list()
    for coordinate in img_coordinates:
        mni_coordinates.append(
            np.array(
                image.coord_transform(
                    coordinate[0],
                    coordinate[1],
                    coordinate[2],
                    img.affine
                )
            )
        )
    print(np.array(mni_coordinates).shape)
    return np.array(mni_coordinates)


@read_or_write('parc')
def parcellation_creating(subject, subjects_dir, labels, _subject_tree=None, _conditions=None):
    vertexes = [
        mne.vertex_to_mni(
            label.vertices,
            hemis=0 if label.hemi == 'lh' else 1,
            subject=subject,
            subjects_dir=subjects_dir
        ) for label in labels
    ]
    return {
        label.name: np.mean(vertex, axis=0) for label, vertex in zip(labels, vertexes)
    }


@read_or_write('con')
def connectivity_computation(label_ts, sfreq, freq_bands, methods, _subject_tree=None, _conditions=None):
    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(freq_bands, list):
        freq_bands = [freq_bands]
    if any([not isinstance(freq_band, tuple) for freq_band in freq_bands]) \
            or any([len(freq_band) % 2 != 0 for freq_band in freq_bands]):
        raise ValueError('freq_bands must contain a list of frequency bands with [minimum_frequency, maximum_frequency]'
                         ' or list of lists with frequency bands, however it contains: {}'.format(freq_bands))

    return {
        str(fmin) + '-' + str(fmax) + 'Hz': {
            method: {
                feature: result
                for feature, result in zip(
                    connectivity_computation_output_features,
                    mne.connectivity.spectral_connectivity(
                        label_ts,
                        method=method,
                        mode='multitaper',
                        sfreq=sfreq,
                        fmin=fmin,
                        fmax=fmax,
                        faverage=True,
                        mt_adaptive=True,
                        n_jobs=1
                    )
                )
            } for method in methods
        } for fmin, fmax in freq_bands
    }


@read_or_write('psd')
def power_spectral_destiny_computation(epochs,
                                       inv,
                                       lambda2,
                                       freq_bands,
                                       method,
                                       bandwidth,
                                       labels,
                                       _subject_tree=None,
                                       _conditions=None):

    def compute_psd_avg(epochs, inv, lambda2, method, fmin, fmax, bandwidth, label):
        psd_stc = mne.minimum_norm.compute_source_psd_epochs(epochs, inv,
                                                   lambda2=lambda2,
                                                   method=method, fmin=fmin, fmax=fmax,
                                                   bandwidth=bandwidth, label=label)
        psd_avg = 0.
        freqs = None
        for i, stc in enumerate(psd_stc):
            if i == 0:
                freqs = stc.times
            psd_avg += stc.data
        psd_avg /= len(epochs)
        return psd_avg, freqs


    if not isinstance(freq_bands, list):
        freq_bands = [freq_bands]
    if any([not isinstance(freq_band, tuple) for freq_band in freq_bands]) \
            or any([len(freq_band) % 2 != 0 for freq_band in freq_bands]):
        raise ValueError('freq_bands must contain a list of frequency bands with [minimum_frequency, maximum_frequency]'
                         ' or list of lists with frequency bands, however it contains: {}'.format(freq_bands))

    return {
        str(fmin) + '-' + str(fmax) + 'Hz': {
            label.name: {
                feature: result
                for feature, result in zip(
                    ('psd', 'freq'),
                    compute_psd_avg(epochs, inv, lambda2, method, fmin, fmax, bandwidth, label)
                )
            } for label in labels
        } for fmin, fmax in freq_bands
    }


@read_or_write('nodes')
def nodes_creation(labels,
                   connectivity,
                   con_methods,
                   psd,
                   nodes_coordinates,
                   resec_coordinates,
                   _subject_tree=None,
                   _conditions=None):

    if not isinstance(con_methods, list):
        con_methods = [con_methods]

    ml_features = con_methods + ['psd']

    def is_resected(node_coordinates, resec_coords):
        for resec_coordinate in resec_coords:
            diff = node_coordinates - resec_coordinate
            dist = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
            if dist <= 1:
                return True
        return False

    nodes = list()
    freq_bands = connectivity.keys()
    if psd.keys() != freq_bands:
        raise ValueError("Connectivity measures and power spectral destiny are computed in different frequency bands")

    for label in labels:
        nodes.append(
            Node(
                label,
                {
                    freq_band: {
                        feature: psd[freq_band][label.name] if feature == 'psd'
                        else connectivity[freq_band][feature][label.name]
                        for feature in ml_features
                    } for freq_band in freq_bands
                },
                nodes_coordinates[label.name],
                'resected' if is_resected(nodes_coordinates[label.name], resec_coordinates) else 'spared'
            )
        )

    if not any([node.type == 'resected' for node in nodes]):
        raise ValueError('Resected nodes not found')

    return nodes



