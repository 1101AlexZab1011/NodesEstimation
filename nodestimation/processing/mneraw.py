import os
import mne
import nibabel
import numpy as np
import nilearn.image as image

from nodestimation.processing import by_default
from nodestimation.processing.connectivity import pearson_ts
from nodestimation.project import read_or_write
from nodestimation import Node


def notchfir(raw, lfreq, nfreq, hfreq):
    # filters the given raw-object from lfreq to nfreq and from nfreq to hfreq

    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    raw_filtered = raw \
        .load_data() \
        .notch_filter(nfreq, meg_picks) \
        .filter(l_freq=lfreq, h_freq=hfreq)

    return raw_filtered


def artifacts_clean(raw, n_components=15, method='correlation', threshold=3.0):
    # makes artifacts cleaning of raw-object using ICA

    ica = mne.preprocessing.ICA(n_components=n_components)
    ica.fit(raw)
    ica.exclude = ica.find_bads_eog(raw)[0] + \
                  ica.find_bads_ecg(raw, method=method, threshold=threshold)[0]

    ica.apply(raw)

    del ica

    return raw


@read_or_write('raw', target='original', write_file=False)
@by_default
def read_original_raw(path, _subject_tree=None, _conditions=None, _priority=0):
    return mne.io.read_raw_fif(path)


@read_or_write('raw', target='nepf')
@by_default
def first_processing(raw, lfreq, nfreq, hfreq,
                     rfreq=None,
                     crop=None,
                     reconstruct=False,
                     meg=True,
                     eeg=True,
                     _subject_tree=None,
                     _conditions=None,
                     _priority=0):
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
@by_default
def bem_computation(subject, subjects_dir, conductivity, _subject_tree=None, _conditions=None, _priority=0):
    model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
    return mne.make_bem_solution(model)


@read_or_write('src')
@by_default
def src_computation(subject, subjects_dir, bem, volume=False, _subject_tree=None, _conditions=None, _priority=0):

    src = mne.setup_source_space(subject, spacing='ico5', add_dist='patch', subjects_dir=subjects_dir)

    if volume:
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

        return src + vol_src

    else:
        return src


@read_or_write('trans', target='original', write_file=False)
@by_default
def read_original_trans(path, _subject_tree=None, _conditions=None, _priority=0):
    return mne.read_trans(path)


@read_or_write('fwd')
@by_default
def forward_computation(info, trans, src, bem, _subject_tree=None, _conditions=None, _priority=0):
    return mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False,
                                     mindist=5.0, n_jobs=1, verbose=True)


@read_or_write('eve')
@by_default
def events_computation(raw, time_points, ids, _subject_tree=None, _conditions=None, _priority=0):
    return np.array([[
        raw.first_samp + raw.time_as_index(time_point)[0],
        0,
        event_id

    ] for time_point, event_id in zip(time_points, ids)])


@read_or_write('epo')
@by_default
def epochs_computation(raw, events, tmin, tmax, _subject_tree=None, _conditions=None, _priority=0):
    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax)


@read_or_write('cov')
@by_default
def noise_covariance_computation(epochs, tmin, tmax, _subject_tree=None, _conditions=None, _priority=0):
    return mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='empirical')


@read_or_write('ave')
@by_default
def evokeds_computation(epochs, _subject_tree=None, _conditions=None, _priority=0):
    return epochs.average()


@read_or_write('inv')
@by_default
def inverse_computation(info, fwd, cov, _subject_tree=None, _conditions=None, _priority=0):
    return mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=None, fixed=False)


@read_or_write('stc')
@by_default
def source_estimation(epochs, inv, lambda2, method, _subject_tree=None, _conditions=None, _priority=0):
    return mne.minimum_norm.apply_inverse_epochs(epochs,
                                                 inv,
                                                 lambda2,
                                                 method,
                                                 pick_ori=None
                                                 )


@read_or_write('coords')
@by_default
def coordinates_computation(subject, subjects_dir, labels, _subject_tree=None, _conditions=None, _priority=0):
    vertexes = [mne.vertex_to_mni(
        label.vertices,
        hemis=0 if label.hemi == 'lh' else 1,
        subject=subject, subjects_dir=subjects_dir
    ) for label in labels]
    return {label.name: np.mean(vertex, axis=0) for label, vertex in zip(labels, vertexes)}


@read_or_write('resec', target='original', write_file=False)
@by_default
def read_original_resec(path, _subject_tree=None, _conditions=None, _priority=0):
    return nibabel.load(path)


@read_or_write('resec_mni')
@by_default
def resection_area_computation(img, _subject_tree=None, _conditions=None, _priority=0):
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


@read_or_write('resec_txt', target='original', write_file=False)
@by_default
def read_original_resec_txt(path, _subject_tree=None, _conditions=None, _priority=0):
    return open(path, 'r').read()


@read_or_write('parc')
@by_default
def parcellation_creating(subject, subjects_dir, labels, _subject_tree=None, _conditions=None, _priority=0):
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


@read_or_write('feat')
@by_default
def features_computation(epochs,
                         inv,
                         lambda2,
                         bandwidth,
                         labels,
                         label_ts,
                         sfreq,
                         freq_bands,
                         methods,
                         se_method,
                         _subject_tree=None,
                         _conditions=None,
                         _priority=0):
    if not isinstance(methods, list):
        methods = [methods]

    def spectral_connectivity_computation(input):
        label_ts, sfreq, fmin, fmax, method = input

        return mne.connectivity.spectral_connectivity(
            label_ts,
            method=method,
            mode='multitaper',
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mt_adaptive=True,
            n_jobs=1
        )[0]

    def power_spectral_destiny_computation(input):
        epochs, inv, lambda2, fmin, fmax, method, bandwidth, labels = input

        def compute_psd_avg(epochs, inv, lambda2, method, fmin, fmax, bandwidth, label):
            psd_stc = mne.minimum_norm.compute_source_psd_epochs(epochs, inv,
                                                                 lambda2=lambda2,
                                                                 method=method, fmin=fmin, fmax=fmax,
                                                                 bandwidth=bandwidth, label=label)
            psd_avg = 0.
            for i, stc in enumerate(psd_stc):
                psd_avg += stc.data
            psd_avg /= len(epochs)
            return psd_avg

        return np.array([
            compute_psd_avg(epochs, inv, lambda2, method, fmin, fmax, bandwidth, label)
            for label in labels
        ])

    def switch_params(epochs,
                      inv,
                      lambda2,
                      bandwidth,
                      labels,
                      label_ts,
                      sfreq,
                      fmin,
                      fmax,
                      method,
                      se_method):

        spectral_connectivity_params = (label_ts, sfreq, fmin, fmax, method)
        psd_params = (epochs, inv, lambda2, fmin, fmax, se_method, bandwidth, labels)

        if method == 'psd':
            return psd_params
        else:
            return spectral_connectivity_params

    out = {
        str(fmin) + '-' + str(fmax) + 'Hz': {
            method: {
                'psd': power_spectral_destiny_computation,
                'coh': spectral_connectivity_computation,
                'cohy': spectral_connectivity_computation,
                'imcoh': spectral_connectivity_computation,
                'plv': spectral_connectivity_computation,
                'ciplv': spectral_connectivity_computation,
                'ppc': spectral_connectivity_computation,
                'pli': spectral_connectivity_computation,
                'pli2_unbiased': spectral_connectivity_computation,
                'wpli': spectral_connectivity_computation,
                'wpli2_debiased': spectral_connectivity_computation,
            }[method](
                switch_params(
                    epochs,
                    inv,
                    lambda2,
                    bandwidth,
                    labels,
                    label_ts,
                    sfreq,
                    fmin,
                    fmax,
                    method,
                    se_method
                )
            )
            for method in methods if method != 'pearson' and method != 'envelope'
        } for fmin, fmax in freq_bands
    }

    if 'pearson' or 'envelope' in methods:
        upd = {'time-domain': {}}
        if 'pearson' in methods:
            upd['time-domain'].update({'pearson': pearson_ts(label_ts)})
        if 'envelope' in methods:
            upd['time-domain'].update({'envelope': mne.connectivity.envelope_correlation(label_ts)})
        out.update(upd)

    return out


@read_or_write('nodes')
@by_default
def nodes_creation(labels,
                   features,
                   nodes_coordinates,
                   resec_coordinates,
                   resec_txt,
                   _subject_tree=None,
                   _conditions=None,
                   _priority=0):

    @by_default
    def is_resected(resec_coordinates, node_coordinates):
        for resec_coordinate in resec_coordinates:
            diff = node_coordinates - resec_coordinate
            dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
            if dist <= 1:
                return True
        return False

    nodes = list()

    for label in labels:
        nodes.append(
            Node(
                label,
                {
                    freq_band: {
                        method: features[freq_band][method][label.name]
                        for method in features[freq_band]
                    } for freq_band in features
                },
                nodes_coordinates[label.name],
                'resected' if is_resected(resec_coordinates, nodes_coordinates[label.name]) else 'spared'
            )
        )

    if resec_txt:
        for node in nodes:
            if node.label.name in resec_txt:
                node.set_type('resected')

    if not any(['resected' in node.type for node in nodes]):
        raise Warning('Resected nodes not found')

    return nodes