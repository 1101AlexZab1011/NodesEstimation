import os
from typing import *

import mne
import nibabel
import numpy as np
import nilearn.image as image
from scipy.fftpack import fftfreq, irfft, rfft
from nodestimation.processing.connectivity import pearson_ts
from nodestimation.project import read_or_write
from nodestimation import Node
from nodestimation.project.annotations import SubjectTree, Features, LabelsFeatures


def notchfir(raw: mne.io.Raw, lfreq: int, nfreq: int, hfreq: int) -> mne.io.Raw:
    """filters the given raw_ object from lfreq to nfreq and from nfreq to hfreq

        :param raw: raw_ to filter
        :type raw: |iraw|_
        :param lfreq: frequency for `low-pass filter <https://en.wikipedia.org/wiki/Low-pass_filter>`_, Hz
        :type lfreq: int
        :param nfreq: frequency for `band-stop filter <https://en.wikipedia.org/wiki/Band-stop_filter>`_, Hz
        :type nfreq: int
        :param hfreq: frequency for `high-pass filter <https://en.wikipedia.org/wiki/High-pass_filter>`_, Hz
        :type hfreq: int
        :return: filtered raw_
        :rtype: mne.io.raw_

        .. _mne.io.raw:
        .. _raw:
        .. _iraw: https://mne.tools/stable/generated/mne.io.Raw.html

        .. |iraw| replace:: *mne.io.raw*
    """

    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    raw_filtered = raw \
        .load_data() \
        .notch_filter(nfreq, meg_picks) \
        .filter(l_freq=lfreq, h_freq=hfreq)

    return raw_filtered


def artifacts_clean(raw: mne.io.Raw, n_components: Optional[Union[int, float, None]] = 15, method: Optional[str] = 'correlation', threshold: Optional[Union[float, str]] = 3.0) -> mne.io.Raw:
    """makes `artifacts <https://www.neuro.mcw.edu/meg/index.php/Artifacts_in_MEG_data>`_ cleaning
        of raw_ object using `ICA <https://en.wikipedia.org/wiki/Independent_component_analysis>`_

        :param raw: raw_ to filter
        :type raw: |iraw|_
        :param n_components: number of principal components (from the pre-whitening PCA step) that are passed to the `ICA algorithm <https://mne.tools/stable/generated/mne.preprocessing.ICA.html>`_ during fitting, default 15
        :type n_components: |iint|_ *or* |ifloat|_ *or* |iNone|_ *, optional*
        :param method: The method used in the `algorithm <https://mne.tools/stable/generated/mne.preprocessing.ICA.html?highlight=ica%20find_bads_ecg#mne.preprocessing.ICA.find_bads_ecg>`_
            for detection `ecg <https://en.wikipedia.org/wiki/Electrocardiography>`_ or `eog <https://en.wikipedia.org/wiki/Electrooculography>`_
            `ICA <https://en.wikipedia.org/wiki/Independent_component_analysis>`_ components, can be either
            "ctps" (`cross-trial phase statistics <https://www.researchgate.net/publication/23302818_Integration_of_Amplitude_and_Phase_Statistics_for_Complete_Artifact_Removal_in_Independent_Components_of_Neuromagnetic_Recordings>`_) or
            `"correlation" <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        :type method: str, optional
        :param threshold: The value above which a feature is classified as outlier, can be computed automatically if given value is "auto", default depends from method (look for
            `original function <https://mne.tools/stable/generated/mne.preprocessing.ICA.html?highlight=ica%20find_bads_ecg#mne.preprocessing.ICA.find_bads_ecg>`_)
        :type threshold: |ifloat|_ *or* |istr|_ *, optional*
        :return: cleaned raw_ object
        :rtype: mne.io.raw_

        .. _iNone: https://docs.python.org/3/library/constants.html#None

        .. |iNone| replace:: *None*
    """

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
        _priority: Optional[int] = None
) -> mne.io.Raw:
    """reads not processed raw_ data, uses :func:`nodestimation.project.read_or_write` decorator

        :param path: path to read
        :type path: str
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: read raw_ object
        :rtype: mne.io.raw_
    """
    return mne.io.read_raw_fif(path)


@read_or_write('raw', search_target='nepf')
def first_processing(
        raw: mne.io.Raw, lfreq: int, nfreq: int, hfreq: int,
        rfreq: Optional[int] = None,
        crop: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        reconstruct: Optional[bool] = False,
        meg: Optional[bool] = True,
        eeg: Optional[bool] = True,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> mne.io.Raw:
    """processes_ given raw_ object, uses :func:`nodestimation.project.read_or_write` decorator

        :param raw: raw_ to filter
        :type raw: |iraw|_
        :param lfreq: frequency for `low-pass filter <https://en.wikipedia.org/wiki/Low-pass_filter>`_, Hz
        :type lfreq: int
        :param nfreq: frequency for `band-stop filter <https://en.wikipedia.org/wiki/Band-stop_filter>`_, Hz
        :type nfreq: int
        :param hfreq: frequency for `high-pass filter <https://en.wikipedia.org/wiki/High-pass_filter>`_, Hz
        :type hfreq: int
        :param crop: end time of the raw data to use (in seconds), if None, does not crop, default None
        :type crop: |ifloat|_ *or* |ilist|_ *of* |ifloat|_ *or* |ituple|_ *of* |ifloat|_ *, optional*
        :param reconstruct: whether make `artifacts <https://www.neuro.mcw.edu/meg/index.php/Artifacts_in_MEG_data>`_ cleaning or not, default False
        :type reconstruct: bool, optional
        :param meg: whether pick meg channel or not, default True
        :type meg: bool, optional
        :param eeg: whether pick meg channel or not, default True
        :type eeg: bool, optional
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: processed raw_ object
        :rtype: mne.io.raw_

        .. _processes:
        .. note:: Processing for raw_ data means
            `band-pass <https://en.wikipedia.org/wiki/Band-pass_filter>`_ and `band-stop <https://en.wikipedia.org/wiki/Band-stop_filter>`_ filtering,
            `cropping <https://mne.tools/stable/generated/mne.io.Raw.html?highlight=raw%20crop#mne.io.Raw.crop>`_ and `artifacts <https://www.neuro.mcw.edu/meg/index.php/Artifacts_in_MEG_data>`_ cleaning
    """

    out = raw.copy()

    if crop:
        if not isinstance(crop, list) or not isinstance(crop, tuple):
            out.crop(tmax=crop)
        elif isinstance(crop, list) \
                or isinstance(crop, tuple) \
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
        _priority: Optional[int] = None
) -> mne.bem.ConductorModel:
    """Computes bem_ solution, uses :func:`nodestimation.project.read_or_write` decorator

        :param subject: patient`s ID
        :type subject: str
        :param subjects_dir: path to directory with patient`s files
        :type subjects_dir: str
        :param conductivity: the conductivities to use for each brain tissue shell. Single element for single-layer model or three elements for three-layer model
        :type conductivity: tuple
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: bem_ solution
        :rtype: mne.bem.ConductorModel_

        .. _imne.bem.ConductorModel:
        .. _mne.bem.ConductorModel:
        .. _bem: https://mne.tools/stable/generated/mne.bem.ConductorModel.html?highlight=conductormodel#mne.bem.ConductorModel
    """
    model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
    return mne.make_bem_solution(model)


@read_or_write('src')
def src_computation(
        subject: str,
        subjects_dir: str,
        bem: mne.bem.ConductorModel,
        volume: Optional[bool] = False,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> Union[mne.SourceSpaces, List[mne.SourceSpaces]]:
    """computes `source spaces`_ solution, uses :func:`nodestimation.project.read_or_write` decorator

        :param subject: patient`s ID
        :type subject: str
        :param subjects_dir: path to directory with patient`s files
        :type subjects_dir: str
        :param bem: bem_ solution
            to build SourceSpaces_
        :type bem: |imne.bem.ConductorModel|_
        :param volume: if True, computes `volume source spaces <https://mne.tools/stable/generated/mne.setup_volume_source_space.html?highlight=setup_volume_source_space#mne.setup_volume_source_space>`_,
            default False
        :type volume: bool, optional
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: `source spaces`_ solution
        :rtype: mne.SourceSpaces_

        .. _imne.SourceSpaces:
        .. _mne.SourceSpaces:
        .. _SourceSpaces:
        .. _`source spaces`: https://mne.tools/stable/generated/mne.SourceSpaces.html#mne.SourceSpaces

        .. |imne.SourceSpaces| replace:: *mne.SourceSpaces*
        .. |imne.bem.ConductorModel| replace:: *mne.bem.ConductorModel*
    """

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
        _priority: Optional[int] = None
) -> Union[dict, List[dict]]:
    """reads given `transformation matrix`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param path: path to read
        :type path: str
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: read `transformation matrix`_
        :rtype: mne.transforms.Transform_

        .. _imne.transforms.Transform:
        .. _mne.transforms.Transform:
        .. _`transformation matrix`: https://mne.tools/stable/generated/mne.transforms.Transform.html#mne.transforms.Transform

        .. |imne.transforms.Transform| replace:: *mne.transforms.Transform*
    """

    return mne.read_trans(path)


@read_or_write('fwd')
def forward_computation(
        info: mne.Info,
        trans: mne.Transform,
        src: mne.SourceSpaces,
        bem: mne.bem.ConductorModel,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> mne.forward.Forward:
    """computes `forward solution`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param info: information about raw data
        :type info: |imne.Info|_
        :param trans: `transformation matrix`_
        :type trans: |imne.transforms.Transform|_
        :param src: `source spaces`_
        :type src: |imne.SourceSpaces|_
        :param bem: bem_ solution
        :type bem: mne.bem.ConductorModel_
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: `forward solution`_
        :rtype: mne.Forward_

        .. _imne.Forward:
        .. _mne.Forward:
        .. _`forward solution`: https://mne.tools/stable/generated/mne.Forward.html?highlight=forward#mne.Forward
        .. _imne.Info: https://mne.tools/stable/generated/mne.Info.html?highlight=info#mne.Info

        .. |imne.Info| replace:: *mne.Info*
    """

    return mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False,
                                     mindist=5.0, n_jobs=1, verbose=True)


@read_or_write('eve')
def events_computation(
        raw: mne.io.Raw,
        time_points: Union[List[int], range, np.ndarray],
        ids: List[int],
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> np.ndarray:
    """creates events for given raw_ object, uses :func:`nodestimation.project.read_or_write` decorator

        :param raw: raw_ to filter
        :type raw: |iraw|_
        :param time_points: times for events start
        :type time_points: |ilist|_ *of* |iint|_ *or* |inp.ndarray|_
        :param ids: number for events identification
        :type ids: |ilist|_ *of* |iint|_
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: array of events
        :rtype: np.ndarray_
    """

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
        _priority: Optional[int] = None
) -> mne.Epochs:
    """creates epochs_ from given raw_ object, uses :func:`nodestimation.project.read_or_write` decorator

            :param raw: raw_ to filter
            :type raw: |iraw|_
            :param events: events to detect epochs
            :type events: |inp.ndarray|_
            :param tmin: start time to an epoch_
            :type tmin: int
            :param tmax: end time to an epoch_
            :type tmax: int
            :param _subject_tree: representation of patient`s files structure, default None
            :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
            :param _priority: if several files are read, which one to choose, if None, read all of them, default None
            :type _priority: int, optional
            :return: array of events
            :rtype: np.ndarray_

            .. _epoched:
            .. _imne.Epochs:
            .. _mne.Epochs:
            .. _epoch:
            .. _epochs: https://mne.tools/stable/python_reference.html?highlight=epoch#module-mne.epochs
        """

    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax)


@read_or_write('cov')
def noise_covariance_computation(
        epochs: mne.Epochs,
        tmin: int,
        tmax: int,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> mne.Covariance:
    """computes `covariance matrix`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param epochs: epochs to compute covariance
        :type epochs: |imne.Epochs|_
        :param tmin: start time to a `covariance matrix`_
        :type tmin: int
        :param tmax: end time to a `covariance matrix`_
        :type tmax: int
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: `covariance matrix`_
        :rtype: mne.Covariance_

        .. _imne.Covariance:
        .. _mne.Covariance:
        .. _`covariance matrix`:
        .. _covariance: https://mne.tools/stable/generated/mne.Covariance.html#mne.Covariance

        .. |imne.Epochs| replace:: *mne.Epochs*
    """

    return mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='empirical')


@read_or_write('ave')
def evokeds_computation(
        epochs: mne.Epochs,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> mne.Evoked:
    """computes evokeds_, uses :func:`nodestimation.project.read_or_write` decorator

        :param epochs: epochs to compute covariance
        :type epochs: |imne.Epochs|_
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: evoked_ data
        :rtype: mne.Evoked_

        .. _mne.Evoked:
        .. _evoked:
        .. _evokeds: https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked
    """

    return epochs.average()


@read_or_write('inv')
def inverse_computation(
        info: mne.Info,
        fwd: mne.Forward,
        cov: mne.Covariance,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> mne.minimum_norm.inverse.InverseOperator:
    """computes `inverse solution`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param info: information about raw data
        :type info: |imne.Info|_
        :param fwd: `forward solution`_
        :type fwd: |imne.Forward|_
        :param cov: `covariance matrix`_
        :type cov: |imne.Covariance|_
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: `source spaces`_ solution
        :rtype: mne.minimum_norm.InverseOperator_

        .. |imne.Covariance| replace:: *mne.Covariance*

        .. |imne.Forward| replace:: *mne.Forward*

        .. _imne.minimum_norm.InverseOperator:
        .. _mne.minimum_norm.InverseOperator:
        .. _`inverse solution`: https://mne.tools/stable/generated/mne.minimum_norm.InverseOperator.html#mne.minimum_norm.InverseOperator
    """
    return mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=None, fixed=False)


@read_or_write('stc')
def source_estimation(
        epochs: mne.Epochs,
        inv: mne.minimum_norm.inverse.InverseOperator,
        lambda2: Union[int, float],
        method: str,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> List[mne.SourceEstimate]:
    """computes `source estimation`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param epochs: epochs to compute covariance
        :type epochs: |imne.Epochs|_
        :param inv: `inverse solution`_
        :type inv: |imne.minimum_norm.InverseOperator|_
        :param lambda2: regularization parameter
        :type lambda2: |iint|_ *or* |ifloat|_
        :param method: `method <nodestimation.html#list-of-mne-solutions>`_ to compute `inverse solution`_
        :type method: str
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: `source estimation`_ for epoched_ data
        :rtype: mne.SourceEstimate_

        .. _mne.SourceEstimate:
        .. _`source estimation`: https://mne.tools/stable/generated/mne.SourceEstimate.html

        .. |imne.minimum_norm.InverseOperator| replace:: *mne.minimum_norm.InverseOperator*
    """
    return mne.minimum_norm.apply_inverse_epochs(
        epochs,
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
        _priority: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """computes central coordinates for given brain regions, uses :func:`nodestimation.project.read_or_write` decorator

        :param subject: patient`s ID
        :type subject: str
        :param subjects_dir: path to directory with patient`s files
        :type subjects_dir: str
        :param labels: brain regions representation
        :type labels: |ilist|_ *of* |imne.Label|_
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: central coordinates relatively to brain region names
        :rtype: |idict|_ *of* |istr|_ *to* |inp.ndarray|_

        .. _labels:
        .. _imne.Label:
        .. _mne.Label: https://mne.tools/stable/generated/mne.Label.html?highlight=label#mne.Label

        .. |imne.Label| replace:: *mne.Label*
    """

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
        _priority: Optional[int] = None
) -> Any:
    """reads given `resection map`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param path: path to read
        :type path: str
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: read `resection map`_
        :rtype: Any

        .. _`resection map`: https://nipy.org/nibabel/reference/nibabel.loadsave.html#module-nibabel.loadsave
    """

    return nibabel.load(path)


@read_or_write('resec_mni')
def resection_area_computation(img: Any, _subject_tree=None, _priority=None):
    """turns given `resection map`_ into mni_ coordinate system, uses :func:`nodestimation.project.read_or_write` decorator

        :param img: `resection map`_
        :type img: Any
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: resection area
        :rtype: np.ndarray_

        .. _mni: https://brainmap.org/training/BrettTransform.html

    """

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
        _priority: Optional[int] = None
) -> str:
    """reads given resection area text description, uses :func:`nodestimation.project.read_or_write` decorator

        :param path: path to read
        :type path: str
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: read resection area text description
        :rtype: str

    """

    return open(path, 'r').read()


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
        _priority: Optional[int] = None
) -> Features:
    """computes `features <nodestimation.html#list-of-metrics>`_, uses :func:`nodestimation.project.read_or_write` decorator

        :param epochs: epochs to compute covariance
        :type epochs: |imne.Epochs|_
        :param inv: `inverse solution`_
        :type inv: |imne.minimum_norm.InverseOperator|_
        :param lambda2: regularization parameter
        :type lambda2: |iint|_ *or* |ifloat|_
        :param bandwidth: the bandwidth of the multi taper windowing function in Hz
        :type bandwidth: |iint|_ *or* |ifloat|_
        :param labels: brain regions representation
        :type labels: |ilist|_ *of* |imne.Label|_
        :param label_ts: time courses for labels_
        :type label_ts: |inp.ndarray|_
        :param sfreq: sample frequency for raw_ object
        :type sfreq: |iint|_ *or* |ifloat|_
        :param freq_bands: frequency bands in which to compute connectivity
        :type freq_bands: |ituple|_ *or* |ilist|_ *of* |ituple|_
        :param methods: `metrics <nodestimation.html#list-of-metrics>`_ to compute
        :type methods: |istr|_ *or* |ilist|_ *of* |istr|_
        :param se_method: metric to compute `mne solution <nodestimation.html#list-of-mne-solutions>`_
        :type se_method: str
        :param _subject_tree: representation of patient`s files structure, default None
        :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
        :param _priority: if several files are read, which one to choose, if None, read all of them, default None
        :type _priority: int, optional
        :return: computed features
        :rtype: look for Features in :mod:`nodestimation.project.annotations`
    """

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

    def correlation_connectivity_computation(input: tuple) -> np.ndarray:
        label_ts, sfreq, fmin, fmax, method = input

        if isinstance(label_ts, list):
            label_ts = np.array(label_ts)

        label_ts = mne.filter.filter_data(label_ts, sfreq, l_freq=fmin, h_freq=fmax, method='fir', copy=True)

        # # filtering
        # dt = 1/sfreq
        # w = fftfreq(label_ts.shape[1], d=dt)
        # filtered_label_ts = label_ts.copy()
        # for i in range(label_ts.shape[0]):
        #     s = filtered_label_ts[i, :]
        #     f_signal = rfft(s)
        #     cut_f_signal = f_signal.copy()
        #     cut_f_signal[(np.abs(w) < fmin)], cut_f_signal[(np.abs(w) > fmax)] = (10**-10, 10**-10)
        #     cs = irfft(cut_f_signal)
        #     filtered_label_ts[i, :] = cs

        return {
            'pearson': pearson_ts,
            'envelope': mne.connectivity.envelope_correlation
        }[method](label_ts)

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

        connectivity_params = (label_ts, sfreq, fmin, fmax, method)
        psd_params = (epochs, inv, lambda2, fmin, fmax, se_method, bandwidth, labels)

        if method == 'psd':
            return psd_params
        else:
            return connectivity_params

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
                'envelope': correlation_connectivity_computation,
                'pearson': correlation_connectivity_computation
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
            for method in methods
        } for fmin, fmax in freq_bands
    }

    return out


@read_or_write('nodes')
def nodes_creation(
        labels: List[mne.Label],
        features: LabelsFeatures,
        nodes_coordinates: np.ndarray,
        resec_coordinates: Union[None, np.ndarray],
        resec_txt: str,
        _subject_tree: Optional[SubjectTree] = None,
        _priority: Optional[int] = None
) -> List[Node]:
    """computes `features <nodestimation.html#list-of-metrics>`_, uses :func:`nodestimation.project.read_or_write` decorator

            :param labels: brain regions representation
            :type labels: |ilist|_ *of* |imne.Label|_
            :param features: `centrality metrics <nodestimation.html#centrality-metrics>`_ values for each `feature <nodestimation.html#list-of-metrics>`_ relatively to labels_
            :type features: *look for LabelsFeatures in* :mod:`nodestimation.project.annotations`
            :param nodes_coordinates: center coordinates in mni_ for each node
            :type nodes_coordinates: |inp.ndarray|_
            :param resec_coordinates: resection area coordinates in mni_ for each node
            :type resec_coordinates: |inp.ndarray|_
            :param resec_txt: resection area text description
            :type resec_txt: str
            :param _subject_tree: representation of patient`s files structure, default None
            :type _subject_tree: *look for SubjectTree in* :mod:`nodestimation.project.annotations` *, optional*
            :param _priority: if several files are read, which one to choose, if None, read all of them, default None
            :type _priority: int, optional
            :return: computed nodes
            :rtype: list_ of :class:`nodestimation.Node`
        """

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

    for label in labels:
        nodes.append(
            Node(
                label,
                {
                    freq_band: {
                        method: {
                            centrality:
                            features[freq_band][method][centrality][label.name]
                            for centrality in features[freq_band][method]
                        } for method in features[freq_band]
                    } for freq_band in features
                },
                nodes_coordinates[label.name],
                'resected' if is_resected(resec_coordinates, nodes_coordinates[label.name]) else 'spared'
            )
        )

    if resec_txt:
        add_resected(resec_txt, nodes)

    if all(['spared' in node.type for node in nodes]):
        for i in range(2, 11):
            print('Resected nodes not found, increase node radius from {} to {}'.format(i-1, i))
            for node in nodes:
                node.type = 'resected' if is_resected(resec_coordinates, node.center_coordinates, i) else 'spared'
            if not all(['spared' in node.type for node in nodes]):
                break

    if all(['spared' in node.type for node in nodes]):
        raise Warning('Resected nodes not found')

    return nodes
