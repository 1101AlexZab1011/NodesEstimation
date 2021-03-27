from abc import ABC
from typing import *

import mne
import os
import pickle
import nodestimation.project as proj
from nodestimation.project.subject import Subject
from nodestimation.project.structures import subject_data_types
from nodestimation.processing.features import \
    prepare_features, \
    prepare_data
from nodestimation.processing.mneraw import \
    read_original_raw, \
    first_processing, \
    src_computation, \
    bem_computation, \
    read_original_trans, \
    forward_computation, \
    events_computation, \
    epochs_computation, \
    evokeds_computation, \
    noise_covariance_computation, \
    inverse_computation, \
    source_estimation, \
    coordinates_computation, \
    read_original_resec, \
    resection_area_computation, \
    features_computation, \
    nodes_creation, \
    read_original_resec_txt


class AbstractPipelineBuffer(ABC):
    def __init__(self, data: Dict[str, Any]):
        for feature in data:
            self.__setattr__(feature, data[feature])

    def __str__(self):
        out = 'PipelineBuffer\n'
        longest_key = len(max(list(self.__dict__.keys()), key=len))
        for key in self.__dict__:
            out += (f'\t{key}:'.ljust(longest_key + 5) + f' {self.__dict__[key]}\n')
        return out

    def __len__(self):
        return len(self.__dict__)

    def __eq__(self, other):
        return self.__dict__.keys() == other.__dict__.keys() and self.__dict__.values() == other.__dict__.values()

    def __lt__(self, other):
        return len(self.__dict__) < len(other.__dict__)

    def __le__(self, other):
        return len(self.__dict__) <= len(other.__dict__)

    def __gt__(self, other):
        return len(self.__dict__) > len(other.__dict__)

    def __ge__(self, other):
        return len(self.__dict__) >= len(other.__dict__)

    def __iter__(self):
        for key in self.__dict__:
            yield self.__dict__[key]

    def __contains__(self, item):
        return item in self.__dict__

    def get_items(self) -> tuple:
        return tuple(self.__dict__.values())


class DefaultPipelineBuffer(AbstractPipelineBuffer):
    def __new__(cls, *args):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DefaultPipelineBuffer, cls).__new__(cls)
        return cls.instance

    def __init__(self, *args):
        if not self.instance.__dict__:
            self.__keys = list(pipeline.__annotations__.keys())[:-2]
            if len(args) < len(self.__keys):
                raise ValueError('Input data too small')
            elif len(args) > len(self.__keys):
                for i in range(len(args) - len(self.__keys)):
                    self.__keys.append(f'unknown_argument_{i}')
            data = {key: value for key, value in zip(self.__keys, args)}
            super().__init__(data)


class PipelineBuffer(AbstractPipelineBuffer):
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        attrs = DefaultPipelineBuffer().__dict__.copy()
        try:
            attrs.pop('_DefaultPipelineBuffer__keys')
        except KeyError:
            print('_DefaultPipelineBuffer__keys is already removed')
        if data is not None:
            attrs.update(data)
        super().__init__(attrs)


def write_subjects(path: str, subjects: List[Subject]) -> None:
    """Stores information for entrie set of :class:`nodestimation.project.subject.Subject` objects

    :param path: path to write
    :type path: str
    :param subjects: list with :class:`nodestimation.project.subject.Subject` objects
    :type subjects: list
    """
    pickle.dump(subjects, open(path, 'wb'))


def read_subjects(path: str) -> List[Subject]:
    """Reads set of :class:`nodestimation.project.subject.Subject` objects

        :param path: path to read
        :type path: str
        :return: list of :class:`nodestimation.project.subject.Subject` objects
        :rtype: list
    """

    return pickle.load(open(path, 'rb'))


def write_subject(path: str, subject: Subject) -> None:
    """Stores information about one :class:`nodestimation.project.subject.Subject` object

        :param path: path to write
        :type path: str
        :param subject: :class:`nodestimation.project.subject.Subject` object
        :type subject: :class:`nodestimation.project.subject.Subject`
    """

    pickle.dump(subject, open(path, 'wb'))


def read_subject(path: str) -> Subject:
    """Reads one :class:`nodestimation.project.subject.Subject` object

            :param path: path to read
            :type path: str
            :rtype: :class:`nodestimation.project.subject.Subject`
        """

    return pickle.load(open(path, 'rb'))


def pipeline(
        crop_time: Optional[int] = 120,
        snr: Optional[float] = 0.5,
        epochs_tmin: Optional[Union[int, float]] = -1,
        epochs_tmax: Optional[Union[int, float]] = 1,
        conductivity: Optional[tuple] = (0.3,),
        se_method: Optional[str] = "sLORETA",
        centrality_metrics: Optional[Union[str, List[str]]] = 'eigen',
        methods: Optional[Union[str, List[str]]] = 'plv',
        rfreq: Optional[int] = 200,
        nfreq: Optional[int] = 50,
        lfreq: Optional[int] = 1,
        hfreq: Optional[int] = 70,
        freq_bands: Optional[Union[tuple, List[tuple]]] = (0.5, 4),
        subjects_specificity: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Subject]:
    """Pipeline for brain data transformation
            **includes:**

        #. notch and bandpass `raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_ filtering
        #. sources signal reconstruction with specified `MNE solution`_
        #. applying specified `processing metric`_
        #. eigencentrality (or integral) computation
        #. results saving

        :param methods: set of metrics to be computed (see `list of metrics`_), default ``"plv"``
        :type methods: |ilist|_ *of* |istr|_ *or* |istr|_, *optional*
        :param centrality_metrics: set of `centrality metrics`_ to compute, default ``"eigen"``
        :type centrality_metrics: |ilist|_ *of* |istr|_ *or* |istr|_, *optional*
        :param se_method: MNE solution for inverse computations (see `list of MNE solutions`_), default ``"sLORETA"``
        :type se_method: str, optional
        :param conductivity: suggested tissues conductivity, default (0.3,)
        :type conductivity: |ituple|_ *of* |ifloat|_, *optional*
        :param epochs_tmin: start time (s) before event, default -1
        :type epochs_tmin: int or float, optional
        :param epochs_tmax: end time (s) after event, default 1
        :type epochs_tmax: int or float, optional
        :param rfreq: resampling frequency (Hz), default 200
        :type rfreq: int, optional
        :param nfreq: frequency (Hz) for notch-filtering, default 50
        :type nfreq: int, optional
        :param lfreq: frequency (Hz) for low-pass-filtering, default 1
        :type lfreq: int, optional
        :param hfreq: frequency (Hz) for high-pass-filtering, default 70
        :type hfreq: int, optional
        :param freq_bands: in what frequency (Hz) diapasons are the calculations performed, default (0.5, 4)
        :type freq_bands: |ituple|_ *of* |ifloat|_ *or* |ilist|_ *of* |ituple|_ *of* |ifloat|_
        :param crop_time: what time (s) of brain data be read, default 200
        :type crop_time: int, optional
        :param snr: regularization parameter, default 0.5
        :type snr: float, optional
        :param subjects_specificity: specific pipeline parameters for exact patients. Have to be a dictionary with patient's IDs as keys and
            dictionaries of pipeline parameters names to value of these parameters as values, if None this parameter does not matter, default None
        :type subjects_specificity: |idict|_ *of* |istr|_ *to* |idict|_ *of* |istr|_ *to Any, optional*
        :return: subjects information, computed according to given parameters
        :rtype: list_ of :class:`nodestimation.project.subject.Subject` objects
        :raise ValueError: if freq_bands given in wrong format

        .. _`processing metric`:
        .. _`list of metrics`:
        .. note:: metrics that can be calculated:
            `psd, coh, cohy, imcoh, plv, ciplv, ppc, pli, pli2_unbiased, wpli,
            wpli2_debiased <https://mne.tools/stable/generated/mne.connectivity.spectral_connectivity.html>`_,
            `pearson <https://www.researchgate.net/figure/nferring-of-Pearson-correlation-based-functional-connectivity-map-including-the-Fishers_fig1_235882165>`_,
            `envelope <https://mne.tools/stable/auto_examples/connectivity/plot_mne_inverse_envelope_correlation.html>`_

        .. _`MNE solution`:
        .. _`list of MNE solutions`:
        .. note:: MNE solutions that can be applied:
            `MNE, dSPM, sLORETA, eLORETA <https://mne.tools/stable/generated/mne.minimum_norm.apply_inverse.html#mne.minimum_norm.apply_inverse>`_

        .. _`centrality metrics`:
        .. note:: Available centrality metrics

            :"degree": `degree centrality <https://en.wikipedia.org/wiki/Centrality#Degree_centrality>`_
            :"eigen": `eigencentrality <https://en.wikipedia.org/wiki/Centrality#Eigenvector_centrality>`_
            :"katz": `katz centrality <https://en.wikipedia.org/wiki/Centrality#Katz_centrality>`_
            :"close": `clossness centrality <https://en.wikipedia.org/wiki/Centrality#Closeness_centrality>`_
            :"harmonic": `harmonic centrality <https://en.wikipedia.org/wiki/Centrality#Harmonic_centrality>`_
            :"between": `betweenness centrality <https://en.wikipedia.org/wiki/Centrality#Betweenness_centrality>`_
            :"info": `information centrality <https://www.sciencedirect.com/science/article/abs/pii/0378873389900166?via%3Dihub>`_

        .. warning:: Be careful choosing centrality metrics. Some connectivity measures are inconsistent with some centrality metrics!

        .. _ifloat: https://docs.python.org/3/library/functions.html#float
        .. _ilist:
        .. _list: https://docs.python.org/3/library/stdtypes.html#list
        .. _ituple: https://docs.python.org/3/library/stdtypes.html#tuple
        .. _istr: https://docs.python.org/3/library/stdtypes.html#str

        .. |ifloat| replace:: *float*
        .. |ilist| replace:: *list*
        .. |ituple| replace:: *tuple*
        .. |istr| replace:: *str*

    """

    def initialize_buffer(subjects_specificity: Dict[str, Dict[str, Any]]) -> Dict[str, PipelineBuffer]:
        return {subject_name: PipelineBuffer(subjects_specificity[subject_name])
                for subject_name in subjects_specificity
                }

    conditions_code = proj.conditions_unique_code(
        crop_time,
        snr,
        epochs_tmin,
        epochs_tmax,
        conductivity,
        se_method,
        methods,
        centrality_metrics,
        rfreq,
        nfreq,
        lfreq,
        hfreq,
        freq_bands
    )
    dbuffer = DefaultPipelineBuffer(
        crop_time,
        snr,
        epochs_tmin,
        epochs_tmax,
        conductivity,
        se_method,
        methods,
        centrality_metrics,
        rfreq,
        nfreq,
        lfreq,
        hfreq,
        freq_bands
    )
    if subjects_specificity is not None:
        sbuffer = initialize_buffer(subjects_specificity)
    else:
        sbuffer = None
    lambda2 = 1.0 / snr ** 2
    subjects_dir, subjects_ = proj.find_subject_dir()
    subjects_file = os.path.join(subjects_dir, 'subjects_information_for_' + conditions_code + '.pkl')
    if os.path.exists(subjects_file):
        print('All computation has been already done, loading of the existing file with the solution...')
        return read_subjects(subjects_file)
    else:
        print('Building of the resources files tree...')
        tree = proj.build_resources_tree(subjects_)
        subjects = list()
        print('Preparing data...')
        for subject_name in tree:
            subject_tree_metadata = tree[subject_name][0]
            subject_file = os.path.join(subject_tree_metadata['path'], 'subject_information_for_' + conditions_code + '.pkl')

            if sbuffer is not None and subject_name in sbuffer:
                crop_time,\
                snr,\
                epochs_tmin,\
                epochs_tmax,\
                conductivity,\
                se_method,\
                methods,\
                centrality_metrics,\
                rfreq,\
                nfreq,\
                lfreq,\
                hfreq,\
                freq_bands = sbuffer[subject_name].get_items()
            else:
                crop_time, \
                snr, \
                epochs_tmin, \
                epochs_tmax, \
                conductivity, \
                se_method, \
                methods, \
                centrality_metrics, \
                rfreq, \
                nfreq, \
                lfreq, \
                hfreq, \
                freq_bands = dbuffer.get_items()

            if not isinstance(methods, list):
                methods = [methods]
            if not isinstance(freq_bands, list):
                freq_bands = [freq_bands]
            if any([not isinstance(freq_band, tuple) for freq_band in freq_bands]) \
                    or any([len(freq_band) % 2 != 0 for freq_band in freq_bands]):
                raise ValueError('freq_bands must contain a list of frequency bands with [minimum_frequency, maximum_frequency]'
                                 ' or list of lists with frequency bands, however it contains: {}'.format(freq_bands))

            if os.path.exists(subject_file):
                print('All computation has been already done, loading of the existing file with the solution...')
                subject = read_subject(subject_file)
            else:
                raw, raw_path = read_original_raw(None, _subject_tree=tree[subject_name], _conditions=None, _priority=0)
                fp_raw, fp_raw_path = first_processing(raw,
                                                       lfreq,
                                                       nfreq,
                                                       hfreq,
                                                       rfreq=rfreq,
                                                       crop=crop_time,
                                                       _subject_tree=tree[subject_name],
                                                       _priority=0)
                bem, bem_path = bem_computation(subject_name,
                                                subjects_dir,
                                                conductivity,
                                                _subject_tree=tree[subject_name],
                                                _priority=0)
                src, src_path = src_computation(subject_name,
                                                subjects_dir,
                                                bem, _subject_tree=tree[subject_name],
                                                _priority=0)
                trans, trans_path = read_original_trans(None, _subject_tree=tree[subject_name], _conditions=None, _priority=0)
                fwd, fwd_path = forward_computation(fp_raw.info,
                                                    trans,
                                                    src,
                                                    bem,
                                                    _subject_tree=tree[subject_name],
                                                    _priority=0)
                eve, eve_path = events_computation(fp_raw,
                                                   range(1, 59),
                                                   [1 for i in range(58)],
                                                   _subject_tree=tree[subject_name],
                                                   _priority=0)
                epo, epo_path = epochs_computation(fp_raw,
                                                   eve,
                                                   epochs_tmin,
                                                   epochs_tmax,
                                                   _subject_tree=tree[subject_name],
                                                   _priority=0)
                cov, cov_path = noise_covariance_computation(epo,
                                                             epochs_tmin,
                                                             0,
                                                             _subject_tree=tree[subject_name],
                                                             _priority=0)
                ave, ave_path = evokeds_computation(epo,
                                                    _subject_tree=tree[subject_name],
                                                    _priority=0)
                inv, inv_path = inverse_computation(epo.info,
                                                    fwd,
                                                    cov,
                                                    _subject_tree=tree[subject_name],
                                                    _priority=0)
                stc, stc_path = source_estimation(epo,
                                                  inv,
                                                  lambda2,
                                                  se_method,
                                                  _subject_tree=tree[subject_name],
                                                  _priority=0)
                resec, resec_path = read_original_resec(None,
                                                        _subject_tree=tree[subject_name],
                                                        _priority=0)
                resec_txt, resec_txt_path = read_original_resec_txt(None,
                                                                    _subject_tree=tree[subject_name],
                                                                    _priority=0)
                resec_mni, resec_mni_path = resection_area_computation(resec,
                                                                       _subject_tree=tree[subject_name],
                                                                       _priority=0)
                labels_parc = mne.read_labels_from_annot(subject_name, parc='aparc.a2009s', subjects_dir=subjects_dir)
                labels_aseg = mne.get_volume_labels_from_src(src, subject_name,
                                                             subjects_dir)
                labels = labels_parc + labels_aseg
                coords, coords_path = coordinates_computation(subject_name,
                                                              subjects_dir,
                                                              labels,
                                                              _subject_tree=tree[subject_name],
                                                              _priority=0)
                label_names = [label.name for label in labels]
                label_ts = mne.extract_label_time_course(stc, labels, src, mode='pca_flip')
                feat, feat_path = features_computation(
                    epo,
                    inv,
                    lambda2,
                    4,
                    labels,
                    label_ts,
                    fp_raw.info['sfreq'],
                    freq_bands,
                    methods,
                    se_method,
                    _subject_tree=tree[subject_name],
                    _priority=0
                )
                nodes, nodes_path = nodes_creation(
                    labels,
                    prepare_features(label_names, feat, centrality_metrics=centrality_metrics),
                    coords,
                    resec_mni,
                    resec_txt,
                    _subject_tree=tree[subject_name],
                    _priority=0
                )
                dataset, dataset_path = prepare_data(
                    nodes,
                    centrality_metrics,
                    _subject_tree=tree[subject_name],
                    _priority=0
                )
                subject = Subject(
                    subject_name,
                    {
                        data_type: data_path
                        for data_type, data_path
                        in zip(
                        subject_data_types, [
                            raw_path,
                            fp_raw_path,
                            bem_path,
                            src_path,
                            trans_path,
                            fwd_path,
                            eve_path,
                            epo_path,
                            cov_path,
                            ave_path,
                            inv_path,
                            stc_path,
                            resec_path,
                            resec_txt_path,
                            resec_mni_path,
                            coords_path,
                            feat_path,
                            nodes_path,
                            dataset_path
                        ]
                    )
                    },
                    nodes,
                    subjects_[subject_name],
                    dataset
                )
                write_subject(subject_file, subject)

            subjects.append(
                subject
            )

        print('All the data has been prepared. Saving the result...')

        write_subjects(subjects_file, subjects)

        print('Successfully saved')

        del dbuffer

        return subjects
