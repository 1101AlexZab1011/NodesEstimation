import os
from typing import *

import mne
import nibabel
import numpy as np
import nilearn.image as image
from nodestimation.processing.connectivity import pearson_ts
from nodestimation.project import read_or_write
from nodestimation import Node
from nodestimation.project.annotations import SubjectTree, Features, LabelsFeatures


def notchfir(raw: mne.io.Raw, lfreq: int, nfreq: int, hfreq: int) -> mne.io.Raw:
    # filters the given raw-object from lfreq to nfreq and from nfreq to hfreq

    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    raw_filtered = raw \
        .load_data() \
        .notch_filter(nfreq, meg_picks) \
        .filter(l_freq=lfreq, h_freq=hfreq)

    return raw_filtered


def artifacts_clean(raw: mne.io.Raw, n_components: Optional[int] = 15, method: Optional[str] = 'correlation', threshold: Optional[float] = 3.0) -> mne.io.Raw:
    # makes artifacts cleaning of raw-object using ICA

    ica = mne.preprocessing.ICA(n_components=n_components)
    ica.fit(raw)
    ica.exclude = ica.find_bads_eog(raw)[0] + \
                  ica.find_bads_ecg(raw, method=method, threshold=threshold)[0]

    ica.apply(raw)

    del ica

    return raw


@read_or_write('raw', search_target='original', write_file=False)
def read_original_raw(
        path: Union[str, None],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.io.Raw:
    return mne.io.read_raw_fif(path)


@read_or_write('raw', search_target='nepf')
def first_processing(
        raw: mne.io.Raw, lfreq: int, nfreq: int, hfreq: int,
        rfreq: Optional[int] = None,
        crop: Optional[float] = None,
        reconstruct: Optional[bool] = False,
        meg: Optional[bool] = True,
        eeg: Optional[bool] = True,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.io.Raw:
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
def bem_computation(
        subject: str,
        subjects_dir: str,
        conductivity: tuple,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.bem.ConductorModel:
    model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
    return mne.make_bem_solution(model)


@read_or_write('src')
def src_computation(
        subject: str,
        subjects_dir: str,
        bem: mne.bem.ConductorModel,
        volume: Optional[bool] = False,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> Union[mne.SourceSpaces, List[mne.SourceSpaces]]:
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


@read_or_write('trans', search_target='original', write_file=False)
def read_original_trans(
        path: Union[str, None],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> Union[dict, List[dict]]:
    return mne.read_trans(path)


@read_or_write('fwd')
def forward_computation(
        info: mne.Info,
        trans: mne.Transform,
        src: mne.SourceSpaces,
        bem: mne.bem.ConductorModel,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.forward.Forward:
    return mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False,
                                     mindist=5.0, n_jobs=1, verbose=True)


@read_or_write('eve')
def events_computation(
        raw: mne.io.Raw,
        time_points: Union[List[int], range, np.ndarray],
        ids: List[int],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> np.ndarray:
    return np.array([[
        raw.first_samp + raw.time_as_index(time_point)[0],
        0,
        event_id

    ] for time_point, event_id in zip(time_points, ids)])


@read_or_write('epo')
def epochs_computation(
        raw: mne.io.Raw,
        events: np.ndarray,
        tmin: int,
        tmax: int,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.Epochs:
    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax)


@read_or_write('cov')
def noise_covariance_computation(
        epochs: mne.Epochs,
        tmin: int,
        tmax: int,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.Covariance:
    return mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='empirical')


@read_or_write('ave')
def evokeds_computation(
        epochs: mne.Epochs,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.Evoked:
    return epochs.average()


@read_or_write('inv')
def inverse_computation(
        info: mne.Info,
        fwd: mne.Forward,
        cov: mne.Covariance,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> mne.minimum_norm.inverse.InverseOperator:
    return mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=None, fixed=False)


@read_or_write('stc')
def source_estimation(
        epochs: mne.Epochs,
        inv: mne.minimum_norm.inverse.InverseOperator,
        lambda2: Union[int, float],
        method: str,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> List[mne.SourceEstimate]:
    return mne.minimum_norm.apply_inverse_epochs(epochs,
                                                 inv,
                                                 lambda2,
                                                 method,
                                                 pick_ori=None
                                                 )


@read_or_write('coords')
def coordinates_computation(
        subject: str,
        subjects_dir: str,
        labels: List[mne.Label],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> Dict[str, np.ndarray]:
    vertexes = [mne.vertex_to_mni(
        label.vertices,
        hemis=0 if label.hemi == 'lh' else 1,
        subject=subject, subjects_dir=subjects_dir
    ) for label in labels]
    return {label.name: np.mean(vertex, axis=0) for label, vertex in zip(labels, vertexes)}


@read_or_write('resec', search_target='original', write_file=False)
def read_original_resec(
        path: Union[str, None],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> Any:
    return nibabel.load(path)


@read_or_write('resec_mni')
def resection_area_computation(img: Any, _subject_tree=None, _conditions=None, _priority=None):
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


@read_or_write('resec_txt', search_target='original', write_file=False)
def read_original_resec_txt(
        path: Union[None, str],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> str:
    return open(path, 'r').read()


@read_or_write('parc')
def parcellation_creating(
        subject: str,
        subjects_dir: str,
        labels: List[mne.Label],
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> Dict[str, np.ndarray]:
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
def features_computation(
        epochs: mne.Epochs,
        inv: mne.minimum_norm.inverse.InverseOperator,
        lambda2: float,
        bandwidth: Union[int, float],
        labels: List[mne.Label],
        label_ts: List[np.ndarray],
        sfreq: Union[int, float],
        freq_bands: Union[tuple, List[tuple]],
        methods: Union[str, List[str]],
        se_method: str,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> Features:
    if not isinstance(methods, list):
        methods = [methods]

    def spectral_connectivity_computation(input: tuple) -> Any:
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

    def power_spectral_destiny_computation(input: tuple) -> np.ndarray:
        epochs, inv, lambda2, fmin, fmax, method, bandwidth, labels = input

        def compute_psd_avg(
                epochs: mne.Epochs,
                inv: mne.minimum_norm.InverseOperator,
                lambda2: float,
                method: Union[str, List[str]],
                fmin: int,
                fmax: int,
                bandwidth: Union[int, float],
                label: mne.Label
        ) -> float:
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

    def switch_params(
            epochs: mne.Epochs,
            inv: mne.minimum_norm.InverseOperator,
            lambda2: float,
            bandwidth: Union[int, float],
            labels: List[mne.Label],
            label_ts: List[np.ndarray],
            sfreq: int,
            fmin: int,
            fmax: int,
            method: Union[str, List[str]],
            se_method: str
    ) -> tuple:

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
def nodes_creation(
        labels: List[mne.Label],
        features: LabelsFeatures,
        nodes_coordinates: np.ndarray,
        resec_coordinates: Union[None, np.ndarray],
        resec_txt: str,
        _subject_tree: Optional[SubjectTree] = None,
        _conditions: Optional[str] = None,
        _priority: Optional[int] = None
) -> List[Node]:

    def is_resected(resec_coordinates: Union[None, np.ndarray], node_coordinates: np.ndarray, radius: int = 1) -> bool:
        if resec_coordinates is not None:
            for resec_coordinate in resec_coordinates:
                diff = node_coordinates - resec_coordinate
                dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
                if dist <= radius:
                    return True
        return False

    def add_resected(resec_txt: str, nodes: List[Node]) -> None:
        for node in nodes:
            if node.label.name in resec_txt:
                node.type = 'resected'

    nodes = list()

    if resec_txt:
    	print(resec_txt)
        add_resected(resec_txt, nodes)

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

    if not any(['resected' in node.type for node in nodes]):
        for i in range(2, 10):
            print('Resected nodes not found, increase node radius from {} to {}'.format(i-1, i))
            for node in nodes:
                node.type = 'resected' if is_resected(resec_coordinates, node.center_coordinates, i) else 'spared'

    if not any(['resected' in node.type for node in nodes]):
        raise Warning('Resected nodes not found')

    return nodes
